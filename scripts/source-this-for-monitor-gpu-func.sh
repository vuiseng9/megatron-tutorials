monitor-gpu () 
{ 
    if ! command -v gpustat &> /dev/null; then
        pip install gpustat;
    fi;
    watch --color -n 1 --no-title gpustat --color
}