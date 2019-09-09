#!/usr/bin/env bash

mkdir test
cd test
wget -O test.zip https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/V6WXAGu2yP7Yhw
unzip test.zip
rm test.zip
cd ..

mkdir val
cd val 
wget -O val.zip https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/zc0jgN3E4xQAdg
unzip val.zip
rm val.zip
cd ..

mkdir train
cd train 
wget -O train.zip https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/btyBnYcMw4iEtw
unzip train.zip
rm train.zip
cd ..


