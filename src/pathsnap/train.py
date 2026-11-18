from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import optuna
from .core import build_dataset, new_model, fit_model, score_rows, save_artifact, iter_trip_dirs, load_trip, CandidateIndex, predict_trip, viterbi

# These five candidate scorers are trained with identical preprocessing so that
# the comparison measures model behaviour rather than different input data.
MODELS=["logistic","naive_bayes","random_forest","hist_gradient","mlp"]

def evaluate_routes(model, root: Path):
    """Score raw per-fix choices and Viterbi-decoded routes on labeled trips."""
    from sklearn.metrics import f1_score
    rows=[]
    for folder in iter_trip_dirs(root):
        trip=load_trip(folder); idx=CandidateIndex(trip); truth=idx.aligned_route(); raw=predict_trip(model,trip)
        decoded=viterbi(trip,raw["candidates"],model)
        continuity=lambda seq: float(np.mean(trip.arcs[np.asarray(seq[:-1]),1] == trip.arcs[np.asarray(seq[1:]),0])) if len(seq)>1 else 1.0
        rows.append({"trip_id":trip.trip_id,"top1_accuracy":float(np.mean(raw["arc_ids"]==truth)),"decoded_accuracy":float(np.mean(decoded==truth)),"macro_f1":float(f1_score(truth,decoded,average="macro",zero_division=0)),"classifier_continuity":continuity(raw["arc_ids"]),"decoded_continuity":continuity(decoded)})
    return {"trips":rows,"mean_top1_accuracy":float(np.mean([r["top1_accuracy"] for r in rows])),"mean_decoded_accuracy":float(np.mean([r["decoded_accuracy"] for r in rows])),"mean_macro_f1":float(np.mean([r["macro_f1"] for r in rows])),"mean_decoded_continuity":float(np.mean([r["decoded_continuity"] for r in rows]))}

def main():
    p=argparse.ArgumentParser(description="Train and compare PathSnap candidate scorers")
    p.add_argument("--data-root",type=Path,required=True,help="Path to extracted Student folder/train")
    p.add_argument("--dev-root",type=Path,help="Optional student_test partition; used only for comparison, never fitting")
    p.add_argument("--out",type=Path,default=Path("artifacts")); p.add_argument("--folds",type=int,default=5); p.add_argument("--trials",type=int,default=12)
    args=p.parse_args(); start=time.time(); x,y,g,trip_groups,miss=build_dataset(args.data_root)
    splits=list(GroupKFold(n_splits=args.folds).split(x,y,groups=trip_groups)); results=[]
    # Cross-validation is grouped by trip to prevent GPS observations from a
    # single route appearing in both a training and validation fold.
    for name in MODELS:
        fold_scores=[]; t0=time.time()
        for tr,va in splits:
            m=fit_model(new_model(name),x[tr],y[tr]); fold_scores.append(score_rows(m,x[va],y[va],g[va]))
        row={"model":name,"cv_pair_macro_f1":float(np.mean([s["pair_macro_f1"] for s in fold_scores])),"cv_log_loss":float(np.mean([s["pair_log_loss"] for s in fold_scores])),"cv_observation_top1":float(np.mean([s["observation_top1"] for s in fold_scores])),"fit_seconds":time.time()-t0}
        results.append(row); print(json.dumps(row))
    winner=max(results,key=lambda r:r["cv_observation_top1"])["model"]
    dev_results={}
    if args.dev_root:
        for row in results:
            m=fit_model(new_model(row["model"]),x,y)
            dev_results[row["model"]]=evaluate_routes(m,args.dev_root)
            row["student_test"]=dev_results[row["model"]]
            print(json.dumps({"model":row["model"],"student_test":dev_results[row["model"]]}))
    tuning={}
    for name in MODELS:
        def objective(trial):
            if name=="logistic":
                params={"C":trial.suggest_float("C",0.05,8,log=True)}
            elif name=="naive_bayes":
                params={"var_smoothing":trial.suggest_float("var_smoothing",1e-12,1e-6,log=True)}
            elif name=="random_forest":
                params={"n_estimators":trial.suggest_int("n_estimators",80,220),"max_depth":trial.suggest_int("max_depth",8,28),"min_samples_leaf":trial.suggest_int("min_samples_leaf",1,8)}
            elif name=="hist_gradient":
                params={"max_iter":trial.suggest_int("max_iter",100,320),"learning_rate":trial.suggest_float("learning_rate",0.02,0.18,log=True),"max_leaf_nodes":trial.suggest_int("max_leaf_nodes",15,63),"l2_regularization":trial.suggest_float("l2_regularization",1e-4,5.0,log=True)}
            else:
                params={"alpha":trial.suggest_float("alpha",1e-5,0.1,log=True),"learning_rate_init":trial.suggest_float("learning_rate_init",1e-4,0.02,log=True),"max_iter":trial.suggest_categorical("max_iter",[80,120,160,250])}
            vals=[]
            for tr,va in splits:
                m=fit_model(new_model(name,params=params),x[tr],y[tr]); vals.append(score_rows(m,x[va],y[va],g[va])["observation_top1"])
            return float(np.mean(vals))
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42)); study.optimize(objective,n_trials=args.trials)
        tuned_model=fit_model(new_model(name,params=study.best_params),x,y)
        tuned_dev=evaluate_routes(tuned_model,args.dev_root) if args.dev_root else None
        tuning[name]={ "best_params":study.best_params,"cv_top1":study.best_value,"student_test":tuned_dev,
                        "trials":[{"number":t.number,"value":t.value,"params":t.params} for t in study.trials]}
        print(json.dumps({"model":name+"_tuned","cv_top1":study.best_value,"best_params":study.best_params,"student_test":tuned_dev}))
    tuned_winner=max(tuning,key=lambda name:tuning[name]["cv_top1"])
    best=fit_model(new_model(tuned_winner,params=tuning[tuned_winner]["best_params"]),x,y)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.dev_root:
        dev_results[tuned_winner+"_tuned"]=evaluate_routes(best,args.dev_root)
    metadata={"selected_model":tuned_winner,"best_params":tuning[tuned_winner]["best_params"],"features":__import__('pathsnap.core',fromlist=['FEATURES']).FEATURES,"models":results,"student_test":dev_results,"tuning":tuning,"cv_top1":tuning[tuned_winner]["cv_top1"],"candidate_miss_rate_train_mean":float(miss.mean()),"training_trips":int(len(np.unique(trip_groups))),"cv_folds":args.folds,"tpe_trials":args.trials,"training_seconds":time.time()-start}
    save_artifact(best,args.out/"pathsnap.joblib",metadata)
    (args.out/"training_results.json").write_text(json.dumps({"models":results,"best_model":tuned_winner,"tuning":tuning,"metadata":metadata},indent=2),encoding="utf-8")
    print(f"Saved {args.out/'pathsnap.joblib'}; tuned_winner={tuned_winner}; CV top1={tuning[tuned_winner]['cv_top1']:.4f}")

if __name__=="__main__": main()
