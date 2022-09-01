if [ "$#" -eq 0  ]; then
    salloc -p 2080ti,gpu --gres=gpu:1 --nodes 16 --ntasks-per-node 1 --mem=6G -c 1
else
    salloc -p 2080ti,gpu --gres=gpu:1 --nodes 16 --ntasks-per-node 1 --mem=10G -c $1
fi


