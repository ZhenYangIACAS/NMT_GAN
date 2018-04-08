export CUDA_VISIBLE_DEVICES='5'
experimentfile=$1
modelFile=$2
sourceDic='/home/user/xushuang1/zy/zhyang/dl4mt/corpus/data_gan_180w_zxw/source_u8.txt.pkl'
targetDic='/home/user/xushuang1/zy/zhyang/dl4mt/corpus/data_gan_180w_zxw/target_u8.txt.pkl'
prefix='NIST02'
sourceFile=${prefix}.seg_u8
targetFile=${prefix}.out
result=${prefix}.result
gpu_device='gpu:0'
var_base=0.10
id=0
histCount = 0

mkdir model_best

while [ 1 ]
do
    id=$(($id+1))
    histCount=$(($histCount+1))

    if [ $histCount -ge 11 ]
    then
        echo "early stopping no improving for 10 times"
        break
    fi
    
    echo "idx: $id"
    cp $experimentfile/${modelFile}.* ./
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

    var=$(perl mteval-v11b.pl -s ${prefix}.xml -r ${prefix}.ref -t "test.txt" -b -d 2|grep 'BLEU score = ')
    var_new=${var:13:7}
    if [ `echo "$var_new > $var_base"|bc` -eq 1 ]
    then
      histCount=0
      var_base=$var_new
      mv ${modelFile}.* ./model_best
      echo "best update in id $id"
    fi
    echo -e "bleu: $var_new \n"
    rm test.txt
    sleep 3600s

done
