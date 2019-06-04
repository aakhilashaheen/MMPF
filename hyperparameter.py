import torch
import torch.nn as nn
from torch.distributions.constraints import positive, simplex

import pyro
import pyro.distributions as dist
from pyro.optim import Adam

from tqdm import tqdm

class BPF(object):

    def __init__(self):
        super().__init__()

    def _model(self, ratings, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        user_mean_dist = dist.Gamma(hyperparams['a_u'], hyperparams['b_u'])
        item_mean_dist = dist.Gamma(hyperparams['a_i'], hyperparams['b_i'])

        with pyro.plate('users_loop', hyperparams['num_users']):
            user_mean = pyro.sample('user_mean', user_mean_dist)
            user_latents_dist = dist.Gamma(hyperparams['c_u'], user_mean)
            with pyro.plate('user_latents', hyperparams['num_latents']):
                user_latents = pyro.sample('user_latents', user_latents_dist)

        with pyro.plate('items_loop', hyperparams['num_items']):
            item_mean = pyro.sample('item_mean', item_mean_dist)
            item_latents_dist = dist.Gamma(hyperparams['c_u'], item_mean)
            with pyro.plate('item_latents',  hyperparams['num_latents']):
                item_latents = pyro.sample('item_latents', item_latents_dist)

        with pyro.plate('ratings',  hyperparams['num_nonmissing']):
            user, item, rating = next(ratings)
            lam = user_latents[:, user] @ item_latents[:, item]
            pyro.sample('obs_rating', dist.Poisson(lam), obs=rating)

    def _guide(self, ratings, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        for u in pyro.plate('users_loop',  hyperparams['num_users']):
            # Variational parameters per user
            q_au = pyro.param('q_au_{}'.format(u), hyperparams['a_u'],
                              constraint=positive)
            q_bu = pyro.param('q_bu_{}'.format(u), hyperparams['b_u'],
                              constraint=positive)
            user_mean_dist = dist.Gamma(q_au, q_bu)
            user_mean = pyro.sample('user_mean', user_mean_dist)

            # Sample latents 
            for l in pyro.plate('user_latents',  hyperparams['num_latents']):
                q_lu1 = pyro.param('q_lu1_{},{}'.format(u, l), hyperparams['c_u'],
                                   constraint=positive)
                q_lu2 = pyro.param('q_lu2_{},{}'.format(u, l), torch.tensor(1.),
                                   constraint=positive)
                user_latents_dist = dist.Gamma(q_lu1, q_lu2)
                user_latents = pyro.sample('user_latents', user_latents_dist)

        for i in pyro.plate('items_loop',  hyperparams['num_items']):
            # Variational parameters per item
            q_ai = pyro.param('q_ai_{}'.format(i), hyperparams['a_i'],
                              constraint=positive)
            q_bi = pyro.param('q_bi_{}'.format(i), hyperparams['b_i'],
                              constraint=positive)
            item_mean_dist = dist.Gamma(q_ai, q_bi)
            item_mean = pyro.sample('item_mean', item_mean_dist)

            # Sample latents 
            for l in pyro.plate('item_latents',  hyperparams['num_latents']):
                q_li1 = pyro.param('q_li1_{},{}'.format(i, l), hyperparams['c_i'],
                                   constraint=positive)
                q_li2 = pyro.param('q_li2_{},{}'.format(i, l), torch.tensor(1.),
                                   constraint=positive)
                item_latents_dist = dist.Gamma(q_li1, q_li2)
                item_latents = pyro.sample('item_latents', item_latents_dist)

    def fit(self, ratings, hyperparams=None, num_steps=2000):
        if hyperparams is None:
            if self.hyperparams is None:
                raise ValueError, 'Hyperparameters not provided!'
            else:
                self.hyperparams = hyperparams
        else:
            hyperparams = self.hyperparams

        svi = SVI(self._model, self._guide, Adam(), loss=TraceGraph_ELBO())
        for s in tqdm(range(num_steps)):
            svi.step(ratings, hyperparams)

class MMPF(BPF):

    def __init__(self):
        super().__init__()

    def _model(self, ratings, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        user_mean_dist = dist.Gamma(hyperparams['a_u'], hyperparams['b_u'])
        item_mean_dist = dist.Gamma(hyperparams['a_i'], hyperparams['b_i'])

        context_dist = dist.Gamma(hyperparams['a_c'], hyperparams['b_c'])
        context_prob_dist = dist.Dirichlet(hyperparams['context_conc'] * torch.ones(hyperparams['num_contexts']))

        user_context_prob = pyro.sample('user_context_prob', context_prob_dist)
        item_context_prob = pyro.sample('item_context_prob', context_prob_dist)
        
        with pyro.plate('user_contexts',  hyperparams['num_contexts']):
            with pyro.plate('user_context_latents',  hyperparams['num_context_latents']):
                user_context_latents = pyro.sample('user_context_latents', context_dist)

        with pyro.plate('item_contexts',  hyperparams['num_contexts']):
            with pyro.plate('item_context_latents',  hyperparams['num_context_latents']):
                item_context_latents = pyro.sample('item_context_latents', context_dist)
                
        with pyro.plate('users_loop',  hyperparams['num_users']):
            user_mean = pyro.sample('user_mean', user_mean_dist)
            user_latents_dist = dist.Gamma(hyperparams['c_u'], user_mean)
            with pyro.plate('user_latents', hyperparams['num_latents']):
                user_latents = pyro.sample('user_latents', user_latents_dist)

        with pyro.plate('items_loop',  hyperparams['num_items']):
            item_mean = pyro.sample('item_mean', item_mean_dist)
            item_latents_dist = dist.Gamma(hyperparams['c_i'], item_mean)
            with pyro.plate('item_latents',  hyperparams['num_latents']):
                item_latents = pyro.sample('item_latents', item_latents_dist)

        for i in pyro.plate('ratings',  hyperparams['num_nonmissing']):
            user, item, rating = next(ratings)
            z_u = pyro.sample('user_context', dist.Categorical(user_context_prob))
            z_i = pyro.sample('item_context', dist.Categorical(item_context_prob))
            lam = user_latents[:, user] @ item_latents[:, item] + \
                user_context_latents[:, z_u] @ item_context_latents[:, z_i]
            pyro.sample('obs_rating', dist.Poisson(lam), obs=rating)

    def _guide(self, ratings, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        q_user_context_conc = pyro.param('q_user_context_conc',
                                         context_conc * torch.ones(hyperparams['num_context_latents']))
        q_item_context_conc = pyro.param('q_item_context_conc',
                                         context_conc * torch.ones(hyperparams['num_context_latents']))
        user_context_prob = pyro.sample('user_context_prob',
                                        dist.Dirichlet(q_user_context_conc))
        item_context_prob = pyro.sample('item_context_prob',
                                        dist.Dirichlet(q_item_context_conc))
        
        with pyro.plate('user_contexts', hyperparams['num_contexts']):
            with pyro.plate('user_context_latents', hyperparams['num_context_latents']):
                user_context_latents = pyro.sample('user_context_latents', context_dist)

        with pyro.plate('item_contexts', hyperparams['num_contexts']):
            with pyro.plate('item_context_latents', hyperparams['num_context_latents']):
                item_context_latents = pyro.sample('item_context_latents', context_dist)

        for u in pyro.plate('users_loop', hyperparams['num_users']):
            # Variational parameters per user
            q_au = pyro.param('q_au_{}'.format(u), hyperparams['a_u'],
                              constraint=positive)
            q_bu = pyro.param('q_bu_{}'.format(u), hyperparams['b_u'],
                              constraint=positive)
            user_mean_dist = dist.Gamma(q_au, q_bu)
            user_mean = pyro.sample('user_mean', user_mean_dist)

            # Sample latents 
            for l in pyro.plate('user_latents', hyperparams['num_latents']):
                q_lu1 = pyro.param('q_lu1_{},{}'.format(u, l), hyperparams['c_u'],
                                   constraint=positive)
                q_lu2 = pyro.param('q_lu2_{},{}'.format(u, l), torch.tensor(1.),
                                   constraint=positive)
                user_latents_dist = dist.Gamma(q_lu1, q_lu2)
                user_latents = pyro.sample('user_latents', user_latents_dist)

        for i in pyro.plate('items_loop', hyperparams['num_items']):
            # Variational parameters per item
            q_ai = pyro.param('q_ai_{}'.format(i), hyperparams['a_i'],
                              constraint=positive)
            q_bi = pyro.param('q_bi_{}'.format(i), hyperparams['b_i'],
                              constraint=positive)
            item_mean_dist = dist.Gamma(q_ai, q_bi)
            item_mean = pyro.sample('item_mean', item_mean_dist)

            # Sample latents 
            for l in pyro.plate('item_latents', hyperparams['num_latents']):
                q_li1 = pyro.param('q_li1_{},{}'.format(i, l), hyperparams['c_i'],
                                   constraint=positive)
                q_li2 = pyro.param('q_li2_{},{}'.format(i, l), torch.tensor(1.),
                                   constraint=positive)
                item_latents_dist = dist.Gamma(q_li1, q_li2)
                item_latents = pyro.sample('item_latents', item_latents_dist)

        for i in pyro.plate('ratings', hyperparams['num_nonmissing']):                   
            user, item, rating = next(ratings)
            q_zu = pyro.param('q_zu_{},{}'.format(user, item), torch.ones(hyperparams['num_contexts']),
                              constraint=positive)
            q_zi = pyro.param('q_zi_{},{}'.format(user, item), torch.ones(hyperparams['num_contexts']),
                              constraint=positive)
            z_u = pyro.sample('user_context', dist.Categorical(q_zu))
            z_i = pyro.sample('item_context', dist.Categorical(q_zi))

    def fit(self, ratings, hyperparams=None, num_steps=2000):
        if hyperparams is None:
            if self.hyperparams is None:
                raise ValueError, 'Hyperparameters not provided!'
            else:
                self.hyperparams = hyperparams
        else:
            hyperparams = self.hyperparams

        svi = SVI(self._model, self._guide, Adam(), loss=TraceGraph_ELBO())
        for s in tqdm(range(num_steps)):
            svi.step(ratings, hyperparamsself, ratings, hyperparams=None)
