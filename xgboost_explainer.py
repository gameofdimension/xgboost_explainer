import xgboost as xgb
# {'tree': 2, 'node': 0, 'is_leaf': False, 'yes': '1', 'no': '2', 'missing': '1', 'gain': '1351.97', 'cover': '1283.28'}
# 12:leaf=0.402452,cover=4.94715
# 2:[f109<-9.53674e-07] yes=5,no=6,missing=5,gain=216.446,cover=712.5

def check_params(tree, eta, lmda):
    right = tree[-1]
    left = tree[-2]
    assert left['is_leaf'] == True
    assert right['is_leaf'] == True
    assert left['parent'] == right['parent']
    parent = tree[left['parent']]

    Hl = left['cover']
    Hr = right['cover']
    Gl = -1.*left['leaf']*(Hl+lmda)/eta
    Gr = -1.*right['leaf']*(Hr+lmda)/eta

    Gp = Gl + Gr
    Hp = Hl + Hr
    expect_gain = Gl**2/(Hl+lmda) + Gr**2/(Hr+lmda) - Gp**2/(Hp+lmda)
    # print(expect_gain, parent['gain'])
    assert abs(expect_gain-parent['gain']) < 1.e-2

def model2table(bst, eta=0.3, lmda=1.0):
    lst_str = bst.get_dump(with_stats=True)
    tree_lst = [[] for _ in lst_str]
    for i,line in enumerate(lst_str):
        # print(i, line)
        tree_idx = i
        parent = {}
        parent[0] = None
        lst_node_str = line.split('\n')
        node_lst = [{} for _ in range(len(lst_node_str)-1)]
        for node in lst_node_str:
            node = node.strip()
            # print("fdfdf",len(node))
            if len(node) <= 0:
                continue
            is_leaf=False
            if ":leaf=" in node:
                is_leaf=True
            # print(segs[0], segs[1])
            node_idx = int(node[:node.index(":")])
            # print(node_idx)
            d = {}
            d['tree'] = tree_idx
            d['node'] = node_idx
            d['is_leaf'] = is_leaf
            if not is_leaf:
                segs = node.split(' ')
                fl = node.index('[')
                fr = node.index('<')
                d['feature'] = node[fl+1:fr]
                for p in segs[1].split(','):
                    k,v = p.split('=')
                    d[k]=v
                d['yes'] = int(d['yes'])
                d['no'] = int(d['no'])
                d['missing'] = int(d['missing'])
                parent[d['yes']] = node_idx
                parent[d['no']] = node_idx
                d['gain'] = float(d['gain'])
                d['cover'] = float(d['cover'])
            else:
                _, lc = node.split(':')
                for p in lc.split(','):
                    k,v = p.split('=')
                    d[k]=v
                d['leaf'] = float(d['leaf'])
                d['cover'] = float(d['cover'])

            # node_lst.append(d)
            node_lst[node_idx] = d
        for j, node in enumerate(node_lst):
            node_lst[j]['parent'] = parent[node_lst[j]['node']]
        tree_lst[i] = node_lst
    for t in tree_lst:
        check_params(t, eta, lmda)
        for j in reversed(range(len(t))):
            node = t[j]
            if node['is_leaf']:
                G = -1.*node['leaf']*(node['cover']+lmda)/eta
            else:
                G = t[node['yes']]['grad'] + t[node['no']]['grad']
            t[j]['grad'] = G
            t[j]['logit'] = -1.*G/(node['cover']+lmda)*eta
    for t in tree_lst:
        for j in reversed(range(len(t))):
            node = t[j]
            if node['parent'] is None:
                node['logit_delta'] = node['logit'] - .0
            else:
                node['logit_delta'] = node['logit'] - t[node['parent']]['logit']

    return tree_lst

def logit_contribution(tree_lst, leaf_lst):
    dist = {'intercept':0.0}
    for i, leaf in enumerate(leaf_lst):
        tree = tree_lst[i]
        node = tree[leaf]
        parent_idx = node['parent']
        # print(node, parent_idx)
        while True:
            if parent_idx is None:
               dist['intercept'] += node['logit_delta'] 
               break
            else:
                parent = tree[parent_idx]
                feat = parent['feature']
                if not feat in dist:
                    dist[feat] = 0.0
                dist[feat] += node['logit_delta']
                node = tree[parent_idx]
                parent_idx = node['parent']
    return dist

