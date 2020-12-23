import argparse
import os

import dgl
import numpy as np

from .graph import KGraph

import code

def kgraph_from_dgl(dgl_graph):
    n_edges = dgl_graph.num_edges()
    relations = dgl_graph.etypes
    rel_ids = {r : i for i, r in enumerate(relations)}

    head = np.empty(n_edges, dtype=np.int32)
    tail = np.empty_like(head)
    relation = np.empty_like(head)

    offset = 0

    for edge_type in dgl_graph.canonical_etypes:
        _, r, _ = edge_type
        h, t = dgl_graph.edges(etype=edge_type)
        
        head[offset : offset + h.size(0)] = h
        tail[offset : offset + h.size(0)] = t
        relation[offset : offset + h.size(0)].fill(rel_ids[r])
        
        offset += h.size(0)


    idx = np.lexsort((tail, relation, head))
    return KGraph.from_htr(head[idx], tail[idx], relation[idx], dgl_graph.num_nodes(), len(relations))


def main():
    parser = argparse.ArgumentParser(description='Import RDF dataset from DGL.')

    # data
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='.')

    args = parser.parse_args()
    

    cls = getattr(dgl.data, args.dataset, None)
    if not issubclass(cls, dgl.data.rdf.RDFGraphDataset):
        raise ValueError('Invalid dataset.')
    
    dataset = cls()

    kgraph = kgraph_from_dgl(dataset[0])
    kgraph.save(os.path.join(args.out_dir, 'graph.npz'))

    data = dataset[0].nodes[dataset.predict_category].data
    
    train_mask = data['train_mask'] == 1
    test_mask = data['test_mask'] == 1

    code.interact(local=locals())

    print(np.min(data['_ID']))
    print(np.max(data['_ID']))
    
    np.savez_compressed(
        os.path.join(args.out_dir, 'labels.npz'),
        train_ids = data['_ID'][train_mask],
        train_labels = data['labels'][train_mask],
        test_ids = data['_ID'][test_mask],
        test_labels = data['labels'][test_mask]
    )
    
    
if __name__ == '__main__':
    main()
