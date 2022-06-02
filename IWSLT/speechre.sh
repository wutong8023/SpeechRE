python input.py

CUDA_VISIBLE_DEVICES=3 fairseq-generate ${DATA_ROOT}     --path ${SAVE_DIR}/speechre_tacred_top5_part_part/ckpts/checkpoint_best.pt     --results-path ${SAVE_DIR}/speechre_tacred_top5_part_part/results/     --user-dir ${IWSLT_ROOT}/fairseq_modules     --task speech_to_text_iwslt21 --gen-subset test_speechre     --max-source-positions 960000 --max-tokens 960000       --skip-invalid-size-inputs-valid-test --prefix-size 1     --beam 5 --scoring sacrebleu

python output.py