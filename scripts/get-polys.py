import sys
from pcas import keyvalue
conn = keyvalue.Connection('ks', address='fanosearch.net:12356')
tab = conn.connect_to_table('reflexive4')
tab.describe() 
import json
polys = {}
for poly in tab.select({'id': 0, 'vertices': ''}, limit=int(sys.argv[1])):
    polys[poly['id']] = json.loads(poly['vertices'])

conn.close()

print(json.dumps(polys))
