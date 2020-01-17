import math

from deepreg.tools.meters import AverageMeter


class Metric:

    all_metrics = dict()

    def __init__(self, name, long_name, meter, best_result_init):
        self.best_result = best_result_init
        self.name = name
        self.long_name = long_name
        self.meter = meter
        Metric.all_metrics[name] = self

    def is_better(self, new_result):
        pass

    def measure(self, loss, pred, target, batch_first, training):
        pass

    def update_meter(self, res, num_items):
        self.meter.update(res, num_items)

    def log_str(self,):
        return "{name} {result.val:.4f} ({result.avg:.4f})".format(name=self.long_name, result=self.meter)

class NegativeLogLikelihoodLoss(Metric):

    def __init__(self):
        super().__init__(
            meter=AverageMeter(),
            long_name='Loss',
            name='loss',
            best_result_init=float('inf'),
        )

    def is_better(self, new_result):
        return new_result < self.best_result

    def measure(self, loss, pred, target, batch_first, training):
        return loss

class Perplexity(Metric):

    def __init__(self):
        super().__init__(
            meter=AverageMeter(),
            long_name='Perplexity',
            name='ppl',
            best_result_init=float('inf'),
        )

    def is_better(self, new_result):
        return new_result < self.best_result

    def measure(self, loss, pred, target, batch_first, training):
        return math.exp(loss)

class Accuracy(Metric):

    def __init__(self):
        super().__init__(
            meter=AverageMeter(),
            long_name='Accurracy',
            name='acc',
            best_result_init=-float('inf'),
        )

    def is_better(self, new_result):
        return new_result > self.best_result

    def measure(self, loss, pred, target, batch_first, training):
        return ((pred == target).long() * (target != PAD).long()).sum()/(target != PAD).long().sum() * 100

class PRF(Metric):

    def __init__(self):
        super().__init__(
            meter=AverageMeter(),
            long_name='F1',
            name='f1',
            best_result_init=-float('inf'),
        )

    def is_better(self, new_result):
        return new_result > self.best_result

    def measure(self, loss, pred, target, batch_first, training):
        tp = ((pred == target).long() * (pred > 0).long() * (target > 0).long()).sum()
        system = (pred > 0).long().sum()
        gold = (target > 0).long().sum()
        P = tp/gold if gold > 0 else 0
        R = tp/system if system > 0 else 0
        # print(P, R, tp, gold, system)
        return 2*P*R/(P+R) * 100 if (P+R) > 0 else 0

[m() for m in [NegativeLogLikelihoodLoss, Perplexity, Accuracy, PRF]]