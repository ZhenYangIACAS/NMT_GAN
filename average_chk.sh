TRAIN_DIR=model3

python avg_checkpoints.py --checkpoints="model.ckpt-109305, model.ckpt-109692, model.ckpt-110050, model.ckpt-110437" \
                          --prefix=$TRAIN_DIR \
                          --output_path="$TRAIN_DIR/averaged.ckpt"
