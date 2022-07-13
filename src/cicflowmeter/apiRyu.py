import requests
import json



class Request:

    def __init__(self):
        self.URL_Switches = "http://localhost:8080/stats/switches"
        self.URL_Flows = "http://localhost:8080/stats/flowentry/add"

    def apiCall(self,dst_ip):
        #print("API CALL")
        r1 = requests.get(url = self.URL_Switches)
        switches = r1.json()
        for dpid in switches:
            match = '"ipv4_dst": "{}", "eth_type": 2048'.format(dst_ip)
            match = ('{%s}' %match)
            pload = '"dpid": {}, "cookie": 1, "cookie_mask":1, "table_id": 0, "idle_timeout": 30, "hard_timeout": 30, "priority": 11111, "match": {}, "actions": []'.format(dpid,match)
            pload = ('{%s}' %(pload))
            r2 = requests.post(url = self.URL_Flows, data = pload)

 
  



#match = '"ipv4_dst": "10.0.0.1", "eth_type": 2048'
#match = ('{%s}' %match)
#pload = '"dpid": 1, "cookie": 1, "cookie_mask":1, "table_id": 0, "idle_timeout": 30, "hard_timeout": 30, "priority": 11111, "match": {}, "actions": []'.format(match)
#pload = ('{%s}' %(pload))
#print(pload) 

# defining a params dict for the parameters to be sent to the API
#PARAMS = {'address':location}
  
# sending get request and saving the response as response object
#r = requests.get(url = URL, params = PARAMS)
#r = requests.post(url = URL2, data = pload) 
#print(r.url) 
# extracting data in json format
#data = r.json()
  
  


  
# printing the output
#print(data)

