set -e

while true
do
    stat=()
    i=0
    for((z = 2; z < 10; z++));
    do
        query=$(gpustat | awk '{print $11}' | sed -n ''${z}'p')
        echo "$query"
        if [ "$query" -lt 100 ]
        then
            stat[$i]=$((z-2))
            i=$((i+1))
        fi
    done
    if [ ${#stat[@]} -ge 3 ]
    then
        echo 'start running'
        time=$(date)
        echo $time
        if [ ${#stat[@]} -lt 3 ]; then
            break
        fi
        #device=${stat[0]},${stat[1]}
        #echo $device
        #gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
        # do something here
        #sleep 2m # 让模型加载入显卡
        export CUDA_VISIBLE_DEVICES=${stat[0]},${stat[1]},${stat[2]}
        echo $CUDA_VISIBLE_DEVICES
        echo "success !"
        #python cli.py --config_yaml classification_ptuning.yaml
        exit 0
        break
    fi
    sleep 2m
done
