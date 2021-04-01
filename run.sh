for _max_grad_norm in 1 0; do
  for _plm_lr in 2e-5 5e-5; do
      for _g_dim in 256 128; do
        for _mult_mask in 1 0; do
          for _g_mult_mask in 1 0; do
            for _lr in 1e-3 1e-4 1e-2; do

              python main_re.py \
                --use_cache 0 \
                --batch_size 4 \
                --num_epoch 15 \
                --grad_accumulation_steps 1 \
                --plm_lr $_plm_lr \
                --lr $_lr \
                --g_dim $_g_dim \
                --patience 5 \
                --max_grad_norm $_max_grad_norm \
                --mult_mask $_mult_mask \
                --g_mult_mask $_g_mult_mask

            done
          done
        done
      done
  done
done
