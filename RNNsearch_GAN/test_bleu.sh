export CUDA_VISIBLE_DEVICES='0'
modelFile='model_best/nmt600w_para_simple_gate'
sourceDic='/home/user/xushuang1/zy/zhyang/dl4mt/corpus/1800w_sents/600w_para/source_u8.txt.pkl'
targetDic='/home/user/xushuang1/zy/zhyang/dl4mt/corpus/1800w_sents/600w_para/target_u8.txt.pkl'
prefix=$1
sourceFile=${prefix}.seg_u8
targetFile=${prefix}.out
result=${prefix}.result
gpu_device='gpu:0'

LD_LIBRARY_PATH="$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/:/home/user/xushuang1/zxw/tools/cuda-8.0/toolkit/lib64" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python` \
          tfGan.py --is_decode True \
                   --saveto $modelFile \
                   --decode_file $sourceFile \
                   --decode_result_file $targetFile \
                   --source_dic $sourceDic  \
                   --target_dic $targetDic  \
                   --decode_gpu $gpu_device \
                   --gpu_device 'gpu-0' \
                   --max_len 50 \
                   --decode_is_print False

perl delEos.pl ${targetFile} ${targetFile}.NoEos
perl after_trans_by_wchen.pl ${targetFile}.NoEos test.txt
perl mteval-v11b.pl -s ${prefix}.xml -r ${prefix}.ref -t "test.txt" -b -d 2 > $result
