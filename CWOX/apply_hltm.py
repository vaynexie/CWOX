# importing the module 
import json 
import os
import pickle
import re

def split1(list1):
    temp2=list1.split(' ')
    temp3=[]
    for aa in temp2:
        bb=int(re.findall(r'\d+', aa)[0])
        if bb>0.9 or re.findall(r'\d+', aa)[0]=='000':temp3.append(bb)
    return temp3

def json_to_dict(json_path):
   with open(json_path) as json_file: 
         data1 = json.load(json_file) 
   node_dict={}
   for h in data1:
       i1=h['children']
       if len(i1)>0:
           temp1=h['id']
           node_dict[temp1]={}

       for i in i1:
           i2=i['children']
           if len(i2)>0:
               temp2=i['id']
               node_dict[temp1][temp2]={}
           if len(i2)==0:
               temp2=i['id']
               if temp2 not in node_dict[temp1]:node_dict[temp1][temp2]={}
               norr=i['text']
               tempp=split1(norr)
               if node_dict[temp1][temp2]=={}:
                   node_dict[temp1][temp2]=tempp
                
           for j in i2:
               i3=j['children']
               if len(i3)>0:
                   temp3=j['id']
                   node_dict[temp1][temp2][temp3]={}
               if len(i3)==0:
                   temp3=j['id']
                   if temp3 not in node_dict[temp1][temp2]:node_dict[temp1][temp2][temp3]={}
                   norr=j['text']
                   tempp=split1(norr)
                   if node_dict[temp1][temp2][temp3]=={}:
                       node_dict[temp1][temp2][temp3]=tempp
               
               for k in i3:
                   i4=k['children']
                   if len(i4)>0:
                       temp4=k['id']
                       node_dict[temp1][temp2][temp3][temp4]={}
                   if len(i4)==0:
                       temp4=k['id']
                       if temp4 not in node_dict[temp1][temp2][temp3]:node_dict[temp1][temp2][temp3][temp4]={}
                       norr=k['text']
                       tempp=split1(norr)
                       if node_dict[temp1][temp2][temp3][temp4]=={}:
                           node_dict[temp1][temp2][temp3][temp4]=tempp
             
                   for l in i4:
                       i5=l['children']
                       if len(i5)>0:
                           temp5=l['id']
                           node_dict[temp1][temp2][temp3][temp4][temp5]={}
                       if len(i5)==0:
                           temp5=l['id']
                           if temp5 not in node_dict[temp1][temp2][temp3][temp4]:
                               node_dict[temp1][temp2][temp3][temp4][temp5]={}
                           norr=l['text']
                           tempp=split1(norr)
                           if node_dict[temp1][temp2][temp3][temp4][temp5]=={}:
                               node_dict[temp1][temp2][temp3][temp4][temp5]=tempp
                    
                       for m in i5:
                           i6=m['children']
                           if len(i6)>0:
                               temp6=m['id']
                               node_dict[temp1][temp2][temp3][temp4][temp5][temp6]={}
                           if len(i6)==0:
                               temp6=m['id']
                               if temp6 not in node_dict[temp1][temp2][temp3][temp4][temp5]:
                                   node_dict[temp1][temp2][temp3][temp4][temp5][temp6]={}
                               norr=m['text']
                               tempp=split1(norr)
                               if node_dict[temp1][temp2][temp3][temp4][temp5][temp6]=={}:
                                   node_dict[temp1][temp2][temp3][temp4][temp5][temp6]=tempp
                               
                           for n in i6:
                               i7=n['children']
                               if len(i7)>0:
                                   temp7=n['id']
                                   node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7]={}
                               if len(i7)==0:
                                   temp7=n['id']
                                   if temp7 not in node_dict[temp1][temp2][temp3][temp4][temp5][temp6]:
                                       node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7]={}  
                                   norr=n['text']
                                   tempp=split1(norr)
                                   if node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7]=={}:
                                       node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7]=tempp
                                   
                               for o in i7:
                                   i8=o['children']
                                   if len(i8)>0:
                                       temp8=o['id']
                                       node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7][temp8]={}
                                   if len(i8)==0:
                                       temp8=o['id']
                                       if temp8 not in node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7]:
                                           node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7][temp8]={}
                                       norr=o['text']
                                       tempp=split1(norr)
                                       if node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7][temp8]=={}:
                                           node_dict[temp1][temp2][temp3][temp4][temp5][temp6][temp7][temp8]=tempp
   return node_dict


def depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0

class apply_hltm:
    def __init__(self,cut_level,json_path):
        self.tree = json_to_dict(json_path)
            
        tree_depth=depth(self.tree)
        tree_depth_range=list(range(1,tree_depth+1))
        cut_depth=tree_depth_range[::-1][cut_level]
        
        # find all pathes in tree to leaf
        def find_all_paths(x, prefix=["root"]):
            if isinstance(x, list):  # height 1 node
                return [[*prefix, val] for val in x]
            else:  # others
                return sum(
                    [find_all_paths(vals, prefix + [key]) for key, vals in x.items()],
                    [],
                )
        self.paths = {path[-1]: path for path in find_all_paths(self.tree)}
        self.cut_depth=cut_depth

    def get_cluster(self, indices):
        cluster = {i: self.paths[i][self.cut_depth] for i in indices}
        indices_map = {c:[] for c in cluster.values()}
        for i, c in cluster.items():
            indices_map[c].append(i)
        return list(indices_map.values())
