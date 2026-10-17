# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) Facebook, Inc. and its affiliates.
# Derived from kNN-box (MIT); modified for ReNovo integration.
# See LICENSE and licenses/ at the repository root.

import torch

def retrieve_k_nearest(query, faiss_index, k):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: i dont know why can't use view but must use reshape here 
    distances, indices = faiss_index.search(
                        query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k)
    
    distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices,device=query.device).view(*query_shape[:-1], k)

    return {"distances": distances, "indices": indices}

class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None


    def retrieve(self, query, return_list = ["vals", "distances"], k = None ):
        r""" 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """

        k = k if k is not None else self.k
        # load the faiss index if haven't loaded
        if not hasattr(self.datastore, "faiss_index") or \
                    self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)

        query = query.detach() 
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], k)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        # other information get from self.datastores.datas using indices, for example `keys` and `vals`
        indices = faiss_results["indices"].cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query"]:
                assert data_name in self.datastore.datas, \
                                    "You must load the {} of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        
        self.results = ret # save the retrieved results
        return ret
    
        