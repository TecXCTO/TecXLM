pkg update && pkg upgrade
pkg install python python-pip clang make cmake ninja libandroid-execinfo libffi openssl

curl -LO https://its-pointless.github.io
bash setup-pointless-repo.sh
pkg install python-torch
