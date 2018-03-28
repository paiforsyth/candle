
ARGFILE=./sh/argfiles/zag_small
RESUME_FILENAME=28_March_2018_Wednesday_16_20_08zag_for_factorize_checkpoint_0
SAVE_PREFIX=zag_svd
SVD_PROP=0.6
REPORT_FILENAME=./reports/zag_factorize_${SVD_PROP}

python -m examples.lab $(cat $ARGFILE) --squeezenet_zag_dont_bypass_last   --res_file=${RESUME_FILENAME} --resume_mode=standard --load_nonstrict --save_prefix=$SAVE_PREFIX --factorize_trained  --proxy_context_type=stdfactorize_context --create_svd_rank_prop=1 --factorize_svd_rank_prop=$SVD_PROP --cuda 


python -m examples.lab $(cat $ARGFILE) --squeezenet_zag_dont_bypass_last   --factorize_use_factors --create_svd_rank_prop=$SVD_PROP --proxy_context_type=stdfactorize_context  --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard    --res_file=${RESUME_FILENAME}_svd_factorize_$SVD_PROP --load_nonstrict  --use_custom_test_data_file --custom_test_data_file="./local_data/cifar10/cifar-10-batches-py/test_batch" --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"

#retrain

python -m examples.lab $(cat $ARGFILE)  --squeezenet_zag_dont_bypass_last  --factorize_use_factors --create_svd_rank_prop=$SVD_PROP --proxy_context_type=stdfactorize_context  --save_prefix=${SAVE_PREFIX}_svd_prop_${SVD_PROP}_retrained  --resume_mode=standard    --res_file=${RESUME_FILENAME}_svd_factorize_$SVD_PROP --load_nonstrict    --cuda
