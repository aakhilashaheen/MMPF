import torch
import torch.nn as nn
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
import pyro.optim


class PoissonFactorization(object):

    def __init__(self):
        super(self, PoissonFactorization).__init__()

    def _model(ratings, num_latents, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        user_mean_dist = dist.Gamma(hyperparams['a_u'], hyperparams['b_u'])
        item_mean_dist = dist.Gamma(hyperparams['a_i'], hyperparams['b_i'])

        with pyro.plate('users_loop', ratings.num_users):
            user_mean = pyro.sample('user_mean', user_mean_dist)
            user_latents_dist = dist.Gamma(hyperparams['c_u'], user_mean)
            with pyro.plate('user_latents', num_latents):
                user_latents = pyro.sample('user_latents', user_latents_dist)

        with pyro.plate('items_loop', ratings.num_items):
            item_mean = pyro.sample('item_mean', item_mean_dist)
            item_latents_dist = dist.Gamma(hyperparams['c_u'], item_mean)
            with pyro.plate('item_latents', num_latents):
                item_latents = pyro.sample('item_latents', item_latents_dist)

        with pyro.plate('ratings', ratings.num_nonmissing):
            user, item, rating = next(ratings)
            lam = user_latents[user] @ item_latents[item]
            pyro.sample('obs_rating', dist.Poisson(lam), obs=rating)

    def _guide(ratings, num_latents, hyperparams=None):
        if hyperparams is None:
            hyperparams = self.hyperparams

        for u in pyro.plate('users_loop', ratings.num_users):
            # Variational parameters per user
            q_au = pyro.param('q_au_{}'.format(u), hyperparams['a_u'])
            q_bu = pyro.param('q_bu_{}'.format(u), hyperparams['b_u'])
            user_mean_dist = dist.Gamma(q_au, q_bu)
            user_mean = pyro.sample('user_mean', user_mean_dist)

            # Sample latents 
            for l in pyro.plate('user_latents', num_latents):
                q_lu1 = pyro.param('q_lu1_{},{}'.format(u, l), hyperparams['c_u'])
                q_lu2 = pyro.param('q_lu2_{},{}'.format(u, l), torch.tensor(1.))
                user_latents_dist = dist.Gamma(q_lu1, q_lu2)
                user_latents = pyro.sample('user_latents', user_latents_dist)

        for i in pyro.plate('items_loop', ratings.num_items):
            # Variational parameters per item
            q_ai = pyro.param('q_ai_{}'.format(i), hyperparams['a_i'])
            q_bi = pyro.param('q_bi_{}'.format(i), hyperparams['b_i'])
            item_mean_dist = dist.Gamma(q_ai, q_bi)
            item_mean = pyro.sample('item_mean', item_mean_dist)

            # Sample latents 
            for l in pyro.plate('item_latents', num_latents):
                q_li1 = pyro.param('q_li1_{},{}'.format(i, l), hyperparams['c_i'])
                q_li2 = pyro.param('q_li2_{},{}'.format(i, l), torch.tensor(1.))
                item_latents_dist = dist.Gamma(q_li1, q_li2)
                item_latents = pyro.sample('item_latents', item_latents_dist)

    def fit(self, ratings, num_latents, hyperparams=None):
        if hyperparams is None:
            if self.hyperparams is None:
                raise ValueError, 'Hyperparameters not provided!'
            else:
                self.hyperparams = hyperparams
        else:
            hyperparams = self.hyperparams

        svi = SVI(self._model, self._optim, 

