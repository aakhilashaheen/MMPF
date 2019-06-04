import torch
import torch.nn as nn
from torch.distributions import Poisson
from torch.distributions.constraints import positive

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate, SVI, Trace_ELBO
from pyro.optim import Adam, Adagrad

# import matplotlib.pyplot as plt

import random

from tqdm import tqdm

class BPF(object):

    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams

    def _model(self, ratings):
        # context_dist = dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c'])
        # context_prob_dist = dist.Dirichlet(self.hyperparams['context_conc'] * torch.ones(self.hyperparams['num_contexts'])) 
        # user_context_prob = pyro.sample('user_context_prob', context_prob_dist)
        # item_context_prob = pyro.sample('item_context_prob', context_prob_dist)
        # 
        user_mean_dist = dist.Gamma(self.hyperparams['a_u'], self.hyperparams['b_u'])
        # user_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])
        # for c in pyro.plate('user_contexts_loop', self.hyperparams['num_contexts']):
        #     for v in pyro.plate('user_context_latents_loop_{}'.format(c),  self.hyperparams['num_context_latents']):
        #         user_context_latents[c, v] = pyro.sample('user_context_latents_{},{}'.format(c, v),
        #                                            dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c']))
        # 
        item_mean_dist = dist.Gamma(self.hyperparams['a_i'], self.hyperparams['b_i'])
        # item_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])
        # for c in pyro.plate('item_contexts_loop', self.hyperparams['num_contexts']):
        #     for v in pyro.plate('item_context_latents_loop_{}'.format(c),  self.hyperparams['num_context_latents']):
        #         item_context_latents[c, v] = pyro.sample('item_context_latents_{},{}'.format(c, v),
        #                                            dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c']))
        # 
        user_mean = torch.empty(self.hyperparams['num_users'])
        user_latents = torch.empty(self.hyperparams['num_users'], self.hyperparams['num_latents'])
        for u in pyro.plate('users_loop', self.hyperparams['num_users']):
            user_mean[u] = pyro.sample('user_mean_{}'.format(u), user_mean_dist)
            user_latents_dist = dist.Gamma(self.hyperparams['c_u'], user_mean[u])
            for k in pyro.plate('user_latents_loop_{}'.format(u), self.hyperparams['num_latents']):
                user_latents[u, k] = pyro.sample('user_latents_{},{}'.format(u, k), user_latents_dist)

        item_mean = torch.empty(self.hyperparams['num_items'])
        item_latents = torch.empty(self.hyperparams['num_items'], self.hyperparams['num_latents'])
        for i in pyro.plate('items_loop', self.hyperparams['num_items']):
            item_mean[i] = pyro.sample('item_mean_{}'.format(i), item_mean_dist)
            item_latents_dist = dist.Gamma(self.hyperparams['c_i'], item_mean[u])
            for k in pyro.plate('item_latents_loop_{}'.format(i), self.hyperparams['num_latents']):
                item_latents[i, k] = pyro.sample('item_latents_{},{}'.format(i, k), item_latents_dist)
                
        ratings_itr = iter(ratings)
        # z_u = torch.empty(self.hyperparams['num_nonmissing'], dtype=torch.long)
        # z_i = torch.empty(self.hyperparams['num_nonmissing'], dtype=torch.long)
        for j in pyro.plate('ratings', self.hyperparams['num_nonmissing']):
            user, item, rating = next(ratings_itr)
            # z_u[j] = pyro.sample('user_context_{},{}'.format(user, item), dist.Categorical(user_context_prob))
            # z_i[j] = pyro.sample('item_context_{},{}'.format(user, item), dist.Categorical(item_context_prob))
            lam = user_latents[user, :] @ item_latents[item, :] # + user_context_latents[z_u[j], :] @ item_context_latents[z_i[j], :]
            pyro.sample('obs_rating_{},{}'.format(user, item), dist.Poisson(lam), obs=rating)

    def _guide(self, ratings):
        # q_user_context_conc = pyro.param('q_user_context_conc', torch.ones(self.hyperparams['num_context_latents']),
        #                                  constraint=positive)
        # q_item_context_conc = pyro.param('q_item_context_conc', torch.ones(self.hyperparams['num_context_latents']),
        #                                  constraint=positive)
        # user_context_prob = pyro.sample('user_context_prob', dist.Dirichlet(q_user_context_conc))
        # item_context_prob = pyro.sample('item_context_prob', dist.Dirichlet(q_item_context_conc))
        # 
        # user_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])        
        # for c in pyro.plate('user_contexts_loop', self.hyperparams['num_contexts']):
        #     for v in pyro.plate('user_context_latents_loop_{}'.format(c), self.hyperparams['num_context_latents']):
        #         q_aucv = pyro.param('q_aucv_{},{}'.format(c, v), self.hyperparams['a_c'],
        #                       constraint=positive)
        #         q_bucv = pyro.param('q_bucv_{},{}'.format(c, v), self.hyperparams['b_c'],
        #                       constraint=positive)
        #         user_context_latents[c, v] = pyro.sample('user_context_latents_{},{}'.format(c, v), dist.Gamma(q_aucv, q_bucv))
        # 
        # item_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])        
        # for c in pyro.plate('item_contexts_loop', self.hyperparams['num_contexts']):
        #     for v in pyro.plate('item_context_latents_loop_{}'.format(c), self.hyperparams['num_context_latents']):
        #         q_aicv = pyro.param('q_aicv_{},{}'.format(c, v), self.hyperparams['a_c'],
        #                       constraint=positive)
        #         q_bicv = pyro.param('q_bicv_{},{}'.format(c, v), self.hyperparams['b_c'],
        #                       constraint=positive)
        #         item_context_latents[c, v] = pyro.sample('item_context_latents_{},{}'.format(c, v), dist.Gamma(q_aicv, q_bicv))

        user_mean = torch.empty(self.hyperparams['num_users'])
        user_latents = torch.empty(self.hyperparams['num_users'], self.hyperparams['num_latents'])
        for u in pyro.plate('users_loop', self.hyperparams['num_users']):
            # Variational parameters per user
            q_au = pyro.param('q_au_{}'.format(u), self.hyperparams['a_u'],
                              constraint=positive)
            q_bu = pyro.param('q_bu_{}'.format(u), self.hyperparams['b_u'],
                              constraint=positive)
            user_mean_dist = dist.Gamma(q_au, q_bu)
            user_mean[u] = pyro.sample('user_mean_{}'.format(u), user_mean_dist)

            # Sample latents
            for l in pyro.plate('user_latents_loop_{}'.format(u), self.hyperparams['num_latents']):
                q_lu1 = pyro.param('q_lu1_{},{}'.format(u, l), self.hyperparams['c_u'],
                                   constraint=positive)
                q_lu2 = pyro.param('q_lu2_{},{}'.format(u, l), torch.tensor(1.),
                                   constraint=positive)
                user_latents_dist = dist.Gamma(q_lu1, q_lu2)
                user_latents[u, l] = pyro.sample('user_latents_{},{}'.format(u, l), user_latents_dist)

        item_mean = torch.empty(self.hyperparams['num_items'])
        item_latents = torch.empty(self.hyperparams['num_items'], self.hyperparams['num_latents'])
        for i in pyro.plate('items_loop', self.hyperparams['num_items']):
            # Variational parameters per item
            q_ai = pyro.param('q_ai_{}'.format(i), self.hyperparams['a_i'],
                              constraint=positive)
            q_bi = pyro.param('q_bi_{}'.format(i), self.hyperparams['b_i'],
                              constraint=positive)
            item_mean_dist = dist.Gamma(q_ai, q_bi)
            item_mean[i] = pyro.sample('item_mean_{}'.format(i), item_mean_dist)

            # Sample latents
            for l in pyro.plate('item_latents_loop_{}'.format(i), self.hyperparams['num_latents']):
                q_li1 = pyro.param('q_li1_{},{}'.format(i, l), self.hyperparams['c_u'],
                                   constraint=positive)
                q_li2 = pyro.param('q_li2_{},{}'.format(i, l), torch.tensor(1.),
                                   constraint=positive)
                item_latents_dist = dist.Gamma(q_li1, q_li2)
                item_latents[i, l] = pyro.sample('item_latents_{},{}'.format(i, l), item_latents_dist)

        # z_u = torch.empty(self.hyperparams['num_nonmissing'])
        # z_i = torch.empty(self.hyperparams['num_nonmissing'])
        # ratings_itr = iter(ratings)
        # for j in pyro.plate('ratings', self.hyperparams['num_nonmissing']):
        #     u, i, r = next(ratings_itr)
        #     q_zu = pyro.param('q_zu_{},{}'.format(u, i), torch.ones(self.hyperparams['num_contexts']),
        #                       constraint=positive)
        #     q_zi = pyro.param('q_zi_{},{}'.format(u, i), torch.ones(self.hyperparams['num_contexts']),
        #                       constraint=positive)
        #     z_u[j] = pyro.sample('user_context_{},{}'.format(u, i), dist.Categorical(q_zu))
        #     z_i[j] = pyro.sample('item_context_{},{}'.format(u, i), dist.Categorical(q_zi))

    def fit(self, ratings, num_steps=2000):
        optim = Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        svi = SVI(self._model, self._guide, optim, loss=Trace_ELBO())
        losses = []
        for s in tqdm(range(num_steps)):
            loss = svi.step(ratings)
            losses.append(loss)

        return losses

class MMPF(object):

    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams

    def _model(self, ratings):
        context_dist = dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c'])
        context_prob_dist = dist.Dirichlet(self.hyperparams['context_conc'] * torch.ones(self.hyperparams['num_contexts'])) 
        user_context_prob = pyro.sample('user_context_prob', context_prob_dist)
        item_context_prob = pyro.sample('item_context_prob', context_prob_dist)

        user_mean_dist = dist.Gamma(self.hyperparams['a_u'], self.hyperparams['b_u'])
        user_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])
        for c in pyro.plate('user_contexts_loop', self.hyperparams['num_contexts']):
            for v in pyro.plate('user_context_latents_loop_{}'.format(c),  self.hyperparams['num_context_latents']):
                user_context_latents[c, v] = pyro.sample('user_context_latents_{},{}'.format(c, v),
                                                   dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c']))
        
        item_mean_dist = dist.Gamma(self.hyperparams['a_i'], self.hyperparams['b_i'])
        item_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])
        for c in pyro.plate('item_contexts_loop', self.hyperparams['num_contexts']):
            for v in pyro.plate('item_context_latents_loop_{}'.format(c),  self.hyperparams['num_context_latents']):
                item_context_latents[c, v] = pyro.sample('item_context_latents_{},{}'.format(c, v),
                                                   dist.Gamma(self.hyperparams['a_c'], self.hyperparams['b_c']))
        
        user_mean = torch.empty(self.hyperparams['num_users'])
        user_latents = torch.empty(self.hyperparams['num_users'], self.hyperparams['num_latents'])
        for u in pyro.plate('users_loop', self.hyperparams['num_users']):
            user_mean[u] = pyro.sample('user_mean_{}'.format(u), user_mean_dist)
            user_latents_dist = dist.Gamma(self.hyperparams['c_u'], user_mean[u])
            for k in pyro.plate('user_latents_loop_{}'.format(u), self.hyperparams['num_latents']):
                user_latents[u, k] = pyro.sample('user_latents_{},{}'.format(u, k), user_latents_dist)

        item_mean = torch.empty(self.hyperparams['num_items'])
        item_latents = torch.empty(self.hyperparams['num_items'], self.hyperparams['num_latents'])
        for i in pyro.plate('items_loop', self.hyperparams['num_items']):
            item_mean[i] = pyro.sample('item_mean_{}'.format(i), item_mean_dist)
            item_latents_dist = dist.Gamma(self.hyperparams['c_i'], item_mean[u])
            for k in pyro.plate('item_latents_loop_{}'.format(i), self.hyperparams['num_latents']):
                item_latents[i, k] = pyro.sample('item_latents_{},{}'.format(i, k), item_latents_dist)
                
        ratings_itr = iter(ratings)
        z_u = torch.empty(self.hyperparams['num_nonmissing'], dtype=torch.long)
        z_i = torch.empty(self.hyperparams['num_nonmissing'], dtype=torch.long)
        for j in pyro.plate('ratings', self.hyperparams['num_nonmissing']):
            user, item, rating = next(ratings_itr)
            z_u[j] = pyro.sample('user_context_{},{}'.format(user, item), dist.Categorical(user_context_prob))
            z_i[j] = pyro.sample('item_context_{},{}'.format(user, item), dist.Categorical(item_context_prob))
            lam = user_latents[user, :] @ item_latents[item, :] + user_context_latents[z_u[j], :] @ item_context_latents[z_i[j], :]
            pyro.sample('obs_rating_{},{}'.format(user, item), dist.Poisson(lam), obs=rating)

    def _guide(self, ratings):
        q_user_context_conc = pyro.param('q_user_context_conc', torch.ones(self.hyperparams['num_context_latents']),
                                         constraint=positive)
        q_item_context_conc = pyro.param('q_item_context_conc', torch.ones(self.hyperparams['num_context_latents']),
                                         constraint=positive)
        user_context_prob = pyro.sample('user_context_prob', dist.Dirichlet(q_user_context_conc))
        item_context_prob = pyro.sample('item_context_prob', dist.Dirichlet(q_item_context_conc))
        
        user_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])        
        for c in pyro.plate('user_contexts_loop', self.hyperparams['num_contexts']):
            for v in pyro.plate('user_context_latents_loop_{}'.format(c), self.hyperparams['num_context_latents']):
                q_aucv = pyro.param('q_aucv_{},{}'.format(c, v), self.hyperparams['a_c'],
                              constraint=positive)
                q_bucv = pyro.param('q_bucv_{},{}'.format(c, v), self.hyperparams['b_c'],
                              constraint=positive)
                user_context_latents[c, v] = pyro.sample('user_context_latents_{},{}'.format(c, v), dist.Gamma(q_aucv, q_bucv))
        
        item_context_latents = torch.empty(self.hyperparams['num_contexts'], self.hyperparams['num_context_latents'])        
        for c in pyro.plate('item_contexts_loop', self.hyperparams['num_contexts']):
            for v in pyro.plate('item_context_latents_loop_{}'.format(c), self.hyperparams['num_context_latents']):
                q_aicv = pyro.param('q_aicv_{},{}'.format(c, v), self.hyperparams['a_c'],
                              constraint=positive)
                q_bicv = pyro.param('q_bicv_{},{}'.format(c, v), self.hyperparams['b_c'],
                              constraint=positive)
                item_context_latents[c, v] = pyro.sample('item_context_latents_{},{}'.format(c, v), dist.Gamma(q_aicv, q_bicv))

        user_mean = torch.empty(self.hyperparams['num_users'])
        user_latents = torch.empty(self.hyperparams['num_users'], self.hyperparams['num_latents'])
        for u in pyro.plate('users_loop', self.hyperparams['num_users']):
            # Variational parameters per user
            q_au = pyro.param('q_au_{}'.format(u), self.hyperparams['a_u'],
                              constraint=positive)
            q_bu = pyro.param('q_bu_{}'.format(u), self.hyperparams['b_u'],
                              constraint=positive)
            user_mean_dist = dist.Gamma(q_au, q_bu)
            user_mean[u] = pyro.sample('user_mean_{}'.format(u), user_mean_dist)

            # Sample latents
            for l in pyro.plate('user_latents_loop_{}'.format(u), self.hyperparams['num_latents']):
                q_lu1 = pyro.param('q_lu1_{},{}'.format(u, l), self.hyperparams['c_u'],
                                   constraint=positive)
                q_lu2 = pyro.param('q_lu2_{},{}'.format(u, l), torch.tensor(1.),
                                   constraint=positive)
                user_latents_dist = dist.Gamma(q_lu1, q_lu2)
                user_latents[u, l] = pyro.sample('user_latents_{},{}'.format(u, l), user_latents_dist)

        item_mean = torch.empty(self.hyperparams['num_items'])
        item_latents = torch.empty(self.hyperparams['num_items'], self.hyperparams['num_latents'])
        for i in pyro.plate('items_loop', self.hyperparams['num_items']):
            # Variational parameters per item
            q_ai = pyro.param('q_ai_{}'.format(i), self.hyperparams['a_i'],
                              constraint=positive)
            q_bi = pyro.param('q_bi_{}'.format(i), self.hyperparams['b_i'],
                              constraint=positive)
            item_mean_dist = dist.Gamma(q_ai, q_bi)
            item_mean[i] = pyro.sample('item_mean_{}'.format(i), item_mean_dist)

            # Sample latents
            for l in pyro.plate('item_latents_loop_{}'.format(i), self.hyperparams['num_latents']):
                q_li1 = pyro.param('q_li1_{},{}'.format(i, l), self.hyperparams['c_u'],
                                   constraint=positive)
                q_li2 = pyro.param('q_li2_{},{}'.format(i, l), torch.tensor(1.),
                                   constraint=positive)
                item_latents_dist = dist.Gamma(q_li1, q_li2)
                item_latents[i, l] = pyro.sample('item_latents_{},{}'.format(i, l), item_latents_dist)

        z_u = torch.empty(self.hyperparams['num_nonmissing'])
        z_i = torch.empty(self.hyperparams['num_nonmissing'])
        ratings_itr = iter(ratings)
        for j in pyro.plate('ratings', self.hyperparams['num_nonmissing']):
            u, i, r = next(ratings_itr)
            q_zu = pyro.param('q_zu_{},{}'.format(u, i), torch.ones(self.hyperparams['num_contexts']),
                              constraint=positive)
            q_zi = pyro.param('q_zi_{},{}'.format(u, i), torch.ones(self.hyperparams['num_contexts']),
                              constraint=positive)
            z_u[j] = pyro.sample('user_context_{},{}'.format(u, i), dist.Categorical(q_zu))
            z_i[j] = pyro.sample('item_context_{},{}'.format(u, i), dist.Categorical(q_zi))

    def fit(self, ratings, num_steps=2000):
        optim = Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        svi = SVI(self._model, self._guide, optim, loss=Trace_ELBO())
        losses = []
        for s in tqdm(range(num_steps)):
            loss = svi.step(ratings)
            losses.append(loss)

        return losses

hyperparams = {}
hyperparams['a_u'] = hyperparams['b_u'] = hyperparams['a_i'] = hyperparams['b_i'] = hyperparams['a_c'] = hyperparams['b_c'] = hyperparams['c_u'] = hyperparams['c_i'] = torch.tensor(1.)
hyperparams['context_conc'] = 5.
hyperparams['num_users'] = 10
hyperparams['num_items'] = 20
hyperparams['num_nonmissing'] = 40
hyperparams['num_latents'] = 4
hyperparams['num_contexts'] = 4
hyperparams['num_context_latents'] = 4

def test(hyperparams)
    idx = [(u, i) for u in range(hyperparam['num_users']) for i in range(hyperparams['num_items'])]
    random.shuffle(idx)
    raw_ratings = Poisson(3.).sample((10, 20))
    ratings = [(u, i, raw_ratings[u, i]) for u, i in idx[:40]]
        
    mmpf = MMPF(hyperparams)
    bpf = BPF(hyperparams)
    mmpf_losses = mmpf.fit(ratings, num_steps=10000)
    bpf_losses = bpf.fit(ratings, num_steps=10000)

    return (bpf_losses, mmpf_losses)
