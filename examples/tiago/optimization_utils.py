import numpy as np
import random
import picos as pc

def rearrange_rb(R_b, param):
    """Rearrange the kinematic regressor by sample numbered order."""
    Rb_rearr = np.empty_like(R_b)
    for i in range(param["calibration_index"]):
        for j in range(param["NbSample"]):
            Rb_rearr[j * param["calibration_index"] + i, :] = R_b[
                i * param["NbSample"] + j
            ]
    return Rb_rearr


def sub_info_matrix(R, param):
    """Return a list of sub information matrices for each data sample."""
    subX_list = []
    idex = param["calibration_index"]
    for it in range(param["NbSample"]):
        sub_R = R[it * idex : (it * idex + idex), :]
        subX = np.matmul(sub_R.T, sub_R)
        subX_list.append(subX)
    return subX_list, dict(zip(np.arange(param["NbSample"]), subX_list))


class Detmax:
    """Determinant maximization algorithm."""

    def __init__(self, candidate_pool, NbChosen):
        self.pool = candidate_pool
        self.nd = NbChosen
        self.cur_set = []
        self.fail_set = []
        self.opt_set = []
        self.opt_critD = []

    def get_critD(self, set):
        """Calculate the n-th squared determinant of information matrix."""
        infor_mat = sum(self.pool[idx] for idx in set)
        return float(pc.DetRootN(infor_mat))

    def main_algo(self):
        """Implement the main algorithm for maximizing the determinant."""
        pool_idx = tuple(self.pool.keys())
        cur_set = random.sample(pool_idx, self.nd)
        updated_pool = list(set(pool_idx) - set(cur_set))

        opt_k = updated_pool[0]
        opt_critD = self.get_critD(cur_set)
        rm_j = cur_set[0]

        while opt_k != rm_j:
            # Add step
            for k in updated_pool:
                cur_set.append(k)
                cur_critD = self.get_critD(cur_set)
                if opt_critD < cur_critD:
                    opt_critD = cur_critD
                    opt_k = k
                cur_set.remove(k)
            cur_set.append(opt_k)
            opt_critD = self.get_critD(cur_set)

            # Remove step
            delta_critD = opt_critD
            rm_j = cur_set[0]
            for j in cur_set:
                rm_set = cur_set.copy()
                rm_set.remove(j)
                cur_delta_critD = opt_critD - self.get_critD(rm_set)
                if cur_delta_critD < delta_critD:
                    delta_critD = cur_delta_critD
                    rm_j = j
            cur_set.remove(rm_j)
            opt_critD = self.get_critD(cur_set)

            self.opt_critD.append(opt_critD)
        return self.opt_critD


class SOCP:
    """Second-Order Cone Programming solver."""

    def __init__(self, subX_dict, param):
        self.pool = subX_dict
        self.param = param
        self.problem = pc.Problem()
        self.w = pc.RealVariable("w", self.param["NbSample"], lower=0)
        self.t = pc.RealVariable("t", 1)

    def add_constraints(self):
        """Add constraints to the optimization problem."""
        Mw = pc.sum(
            self.w[i] * self.pool[i] for i in range(self.param["NbSample"])
        )
        self.problem.add_constraint(1 | self.w <= 1)
        self.problem.add_constraint(self.t <= pc.DetRootN(Mw))

    def set_objective(self):
        """Set the objective function for optimization."""
        self.problem.set_objective("max", self.t)

    def solve(self):
        """Solve the optimization problem."""
        self.add_constraints()
        self.set_objective()
        self.solution = self.problem.solve(solver="cvxopt")

        w_list = [float(self.w.value[i]) for i in range(self.w.dim)]
        print("Sum of all elements in vector solution: ", sum(w_list))

        w_dict = dict(zip(np.arange(self.param["NbSample"]), w_list))
        w_dict_sort = dict(
            reversed(sorted(w_dict.items(), key=lambda item: item[1]))
        )
        return w_list, w_dict_sort
