### Collection of helpful rebuild commands:

```
# Stop services
docker-compose down

# Rebuild with new requirements
docker-compose build

# Restart
docker-compose up -d
```


#### Update Local Environment - Conda/Mamba:
```
# Deactivate and remove old
conda deactivate
mamba env remove -n airflow_qdrant_lab

# Create fresh with new versions
mamba env create -f env/airflow_qdrant_lab.yml
conda activate airflow_qdrant_lab

# Verify
python -c "import qdrant_client; print(qdrant_client.__version__)"
```


```
jupyter kernelspec list

jupyter kernelspec remove airflow_qdrant_lab
jupyter kernelspec remove name

python -m ipykernel install --user --name airflow_qdrant_lab --display-name "Airflow Qdrant Lab2"
```

### Sometimes, qdrant may crash due to incompatible data files - to reset:

```
# Stop services
docker-compose down

# Remove ALL qdrant storage (including the problematic collection)
Remove-Item -Recurse -Force .\qdrant_storage\*

# Verify it's empty
dir .\qdrant_storage\

# Start fresh
docker-compose up -d
```
