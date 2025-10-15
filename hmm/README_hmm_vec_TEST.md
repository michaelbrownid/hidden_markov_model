================================
# Goal

Goal: tests for the hmm_vec code

================================
# TEST_hmm_vec_train: Training for hmm_vec

Set up single input

##### Get inputs / output
/media/datastore/storestudio/enterprise/ai_video/holly-experiments-mb-2025Q3/redbullCohesive/redbull_metadata/RB_06.json.embeddings \
/media/datastore/storestudio/enterprise/ai_video/holly-experiments-mb-2025Q3/lumeno_style_redbull/RB_model.json , RB_MODEL_NEWNEW.GOLD.json

##### run the estimation

conda activate basecatch

export PYTHONPATH="/media/datastore/storestudio/workspace2022/hidden-markov-model"

python3 -m hmm.HMM_VEC_TRAIN_ACC \
--model TESTGOLD_hmm_vec_train_model.json \
--newmodel TEST_hmm_vec_train_model_NEW.json \
--dataListTSV <(echo TESTGOLD_hmm_vec_train_input.data) \
> TEST_hmm_vec_train.cout 2> TEST_hmm_vec_train.cerr

Compute diff between models

python3 -m hmm.HMM_VEC_DIFF \
--model TEST_hmm_vec_train_model_NEW.json \
--goldmodel TESTGOLD_hmm_vec_train_model_NEW.json \
> TEST_hmm_vec_train.diff.cout

diffmax HMMState 6 0.0
diffmax HMMState 7 0.0
diffmax HMMState 8 0.0
diffmax HMMState 9 0.0
diffmax HMMState 10 0.0
diffmax HMMState 11 0.0
diffmax HMMState 12 0.0
diffmax mean HMMStateOutputMVN 0 0.0
diffmax sd HMMStateOutputMVN 0 0.0
diffmax mean HMMStateOutputMVN 1 0.0
diffmax sd HMMStateOutputMVN 1 0.0
diffmax mean HMMStateOutputMVN 2 0.0
diffmax sd HMMStateOutputMVN 2 0.0
diffmax mean HMMStateOutputMVN 3 0.0
diffmax sd HMMStateOutputMVN 3 0.0
diffmax mean HMMStateOutputMVN 4 0.0
diffmax sd HMMStateOutputMVN 4 0.0
diffmax mean HMMStateOutputMVN 5 0.0
diffmax sd HMMStateOutputMVN 5 0.0

RESULT: all model diffs against GOLD are zero. Exactly the same
estimated model. Also tests mean/variance sums vs sumsOfSquares
estimation.

Compare to previous log files
diff TEST_hmm_vec_train.cerr TESTGOLD_hmm_vec_train.cerr
diff TEST_hmm_vec_train.cout TESTGOLD_hmm_vec_train.cout
diff TEST_hmm_vec_train.diff.cout TESTGOLD_hmm_vec_train.diff.cout

================================
