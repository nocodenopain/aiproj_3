import time
import functorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os

P = 3  # spline degree
N_CTPS = 5  # number of control points

RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256


def _splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    """
        x: (t,)
        t: (m, )
        c: (n_ctps, dim)
    """
    assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m= n + k + 1

    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(0)
    u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u - k + torch.arange(k + 1, device=c.device)].contiguous()
    for r in range(1, k + 1):
        j = torch.arange(r - 1, k, device=c.device) + 1
        t0 = t[j + u - k]
        t1 = t[j + u + 1 - r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
    return d[:, k]


def splev(
        x: torch.Tensor,
        knots: torch.Tensor,
        ctps: torch.Tensor,
        degree: int,
        der: int = 0
) -> torch.Tensor:
    """Evaluate a B-spline or its derivatives.

    See https://en.wikipedia.org/wiki/B-spline for more about B-Splines.
    This is a PyTorch implementation of https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

    Parameters
    ----------
    x : Tensor of shape `(t,)`
        An array of points at which to return the value of the smoothed
        spline or its derivatives.
    knots: Tensor of shape `(m,)`
        A B-Spline is a piece-wise polynomial.
        The values of x where the pieces of polynomial meet are known as knots.
    ctps: Tensor of shape `(n_ctps, dim)`
        Control points of the spline.
    degree: int
        Degree of the spline.
    der: int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    """
    if der == 0:
        return _splev_torch_impl(x, knots, ctps, degree)
    else:
        assert der <= degree, "The order of derivative to compute must be less than or equal to k."
        n = ctps.size(-2)
        ctps = (ctps[..., 1:, :] - ctps[..., :-1, :]) / (knots[degree + 1:degree + n] - knots[1:n]).unsqueeze(-1)
        return degree * splev(x, knots[..., 1:-1], ctps, degree - 1, der - 1)


def compute_traj(ctps_inter: torch.Tensor):
    """Compute the discretized trajectory given the second to the second control points"""
    t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device),
        torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
        torch.full((P,), N_CTPS - P, device=ctps_inter.device),
    ])
    ctps = torch.cat([
        torch.tensor([[0., 0.]], device=ctps_inter.device),
        ctps_inter,
        torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
    ])
    return splev(t, knots, ctps, P)


def evaluate(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
) -> torch.Tensor:
    """Evaluate the trajectory and return the score it gets.

    Parameters
    ----------
    traj: Tensor of shape `(*, T, 2)`
        The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
    target_pos: Tensor of shape `(N, 2)`
        x-y positions of shape where `N` is the number of targets.
    target_scores: Tensor of shape `(N,)`
        Scores you get when the corresponding targets get hit.
    """
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=89)
        self.fc2 = nn.Linear(in_features=89, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    # def predict(self, x):
    #     pred = F.softmax(self.forward(x))
    #     ans = []
    #     for t in pred:
    #         if t[0] > t[1]:
    #             ans.append(0)
    #         else:
    #             ans.append(1)
    #     return torch.tensor(ans)


class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        # TODO: prepare your agent here
        # data = torch.load('data.pth')
        # feature = data['feature']
        # label = data['label']
        # # X = torch.from_numpy(feature).type(torch.FloatTensor)
        # # y = torch.from_numpy(label).type(torch.LongTensor)
        # X = feature
        # y = label
        # model = MyClassifier()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # epochs = 50000
        # losses = []
        # for i in range(epochs):
        #     # Precit the output for Given input
        #     y_pred = model.forward(X)
        #     # Compute Cross entropy loss
        #     loss = criterion(y_pred, y)
        #     # Add loss to the list
        #     losses.append(loss.item())
        #     # Clear the previous gradients
        #     optimizer.zero_grad()
        #     # Compute gradients
        #     loss.backward()
        #     # Adjust weights
        #     optimizer.step()
        # torch.save(model.state_dict(), 'model.pth')
        # self.model = model
        self.func = None
        self.model = MyClassifier()
        TEST_FILENAME = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.model.load_state_dict(torch.load(TEST_FILENAME))
        self.model.eval()

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Compute the parameters required to fire a projectile.
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """

        def evaluate_r(rand):
            return evaluate(compute_traj(rand), target_pos, class_scores[cls], RADIUS)

        assert len(target_pos) == len(target_features)
        # TODO: compute the firing speed and angle that would give the best score.
        pred = F.softmax(self.model.forward(target_features))
        cls = pred.argmax(dim=1)
        self.func = functorch.vmap(evaluate_r)
        # Example: return a random configuration
        ctps_inter = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        # ctps_inter.requires_grad = True
        # t = time.time()
        # global val
        # lr = 1
        # while time.time() - t < 0.03:
        #     # read_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[r], RADIUS)
        #     gra_score = evaluate_plus(compute_traj(ctps_inter), target_pos, class_scores[cls], RADIUS)
        #     # print(it, read_score.item())
        #     gra_score.backward()
        #     ctps_inter.data = ctps_inter.data + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
        # optimizer = torch.optim.Adam(params=[-ctps_inter.data], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
        #                              amsgrad=False)
        # for i in range(9):
        #     pred = evaluate_plus(compute_traj(ctps_inter), target_pos, class_scores[cls], RADIUS)
        #     optimizer.zero_grad()
        #     pred.backward()
        #     optimizer.step()

        val = evaluate(compute_traj(ctps_inter), target_pos, class_scores[cls], RADIUS)
        t = time.time()
        # while time.time() - t < 0.26:
        #     r = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        #     tmp = evaluate(compute_traj(r), target_pos, class_scores[cls], RADIUS)
        #     if tmp > val:
        #         val = tmp
        #         ctps_inter = r
        while time.time() - t < 0.25:
            r = torch.rand((1000, N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            tmp = self.func(r)
            index = tmp.argmax()
            tmp_score = tmp[index]
            if tmp_score > val:
                val = tmp_score
                ctps_inter = r[index]
        # t = time.time()
        # ctps_inter.requires_grad = True
        # while time.time() - t < 0.03:
        #     # read_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[r], RADIUS)
        #     gra_score = evaluate_plus(compute_traj(ctps_inter), target_pos, class_scores[cls], RADIUS)
        #     # print(it, read_score.item())
        #     gra_score.backward()
        #     ctps_inter.data = ctps_inter.data + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
        return ctps_inter
