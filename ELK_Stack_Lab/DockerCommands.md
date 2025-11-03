## Docker Commands


#### Navigate to Docker folder
cd ELK_Stack_Lab\ELK_Docker
(or)
cd ELK_Docker

#### Pull images (this downloads ~2GB, takes 5-10 min)
docker-compose pull

#### Start the stack
docker-compose up -d

#### Wait 2-3 minutes for services to start

#### Check status
docker-compose ps
#### All 3 should show "Up" and healthy

#### Check logs (if something fails)
docker-compose logs elasticsearch
docker-compose logs logstash
docker-compose logs kibana

#### Verify Elasticsearch is working
curl http://localhost:9200
#### Should return JSON with cluster info

-----------------------------------------------------------------------------------------------------------------------

#### Verify Kibana is working (open browser)
- http://localhost:5601

#### View Logstash logs
docker logs fraud-logstash

#### List all indices
curl http://localhost:9200/_cat/indices?v

- for example, this returns (  {"count":25,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}} ) because i have 25 logs right now !

#### 1. Check Logstash logs
docker logs fraud-logstash --tail 30

#### 2. Check if index exists
curl http://localhost:9200/_cat/indices?v

#### 3. Check document count
curl http://localhost:9200/fraud-monitoring-*/_count


#### To restart logstash:

cd ELK_Docker
docker-compose restart logstash

-----------------------------------------------------------------------------------------------------------------------
#### Some other basic check commands.

docker --version
# Should show: Docker version 24.x.x...

docker --version
docker ps


#### Final shutdown or close commands:

#### Stop all containers gracefully (keeps volumes/data)
docker-compose stop

#### Or stop + remove containers (but KEEP volumes/data)
docker-compose down

### See running containers
docker-compose ps

### See all containers (including stopped)
docker ps -a