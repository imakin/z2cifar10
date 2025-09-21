# python s1model.py --name mc2hc3 --filter0 12 --filter1 8 --filter2 8 --filter3 6 --prune3 0.2
# python s2hlsmodelZ2.py --sambung --input=keras/mc2hc3/main_full.keras --output=mc2hc3 --minim

# python s1model.py --name mc2hc4 --filter0 12 --filter1 8 --filter2 8 --filter3 6 --prune3 0.4
# python s2hlsmodelZ2.py --sambung --input=keras/mc2hc3/main_full.keras --output=mc2hc4 --minim

# python s1model.py --name mc2hc5 --filter0 10 --filter1 8 --filter2 6 --filter3 6 --prune3 0.2
# python s2hlsmodelZ2.py --sambung --input=keras/mc2hc5/main_full.keras --output=mc2hc5 --minim

# python s1model.py --name mc2hc6 --filter0 10 --filter1 8 --filter2 6 --filter3 6 --prune3 0.4
# python s2hlsmodelZ2.py --sambung --input=keras/mc2hc6/main_full.keras --output=mc2hc6 --minim

python s1model.py --name mc2hc7 --filter0 8 --filter1 6 --filter2 6 --filter3 6 --prune2 0.2 --prune3 0.4
python s2hlsmodelZ2.py --sambung --input=keras/mc2hc7/main_full.keras --output=mc2hc7 --minim

python s1model.py --name mc2hc8 --filter0 8 --filter1 6 --filter2 6 --filter3 6 --prune2 0.5 --prune3 0.5
python s2hlsmodelZ2.py --sambung --input=keras/mc2hc8/main_full.keras --output=mc2hc8 --minim

python s1model.py --name mc2hc9 --filter0 8 --filter1 6 --filter2 3 --filter3 6 --prune2 0.5 --prune3 0.5
python s2hlsmodelZ2.py --sambung --input=keras/mc2hc9/main_full.keras --output=mc2hc9 --minim

