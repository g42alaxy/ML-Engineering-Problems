### To create docker container

docker build -t wordtools:latest .

### To run a container 

docker run -d --name mycontainer wordtools:latest tail -f /dev/null

### To execute all scripts and look up for their results 
docker exec mycontainer /usr/local/bin/word_count.sh /data/dracula.txt | head -n 20
docker exec mycontainer /usr/local/bin/top10_create_files.sh /data/dracula.txt /data/out

### Output folder content: 

docker exec mycontainer ls -l  /data/out