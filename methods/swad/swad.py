import os
import torch
import time
import datetime
import copy
from collections import deque
import numpy as np
from . import swa_utils
import pdb
from ..base_trainer import BaseTrainer
import torch.nn.functional as F
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..utils import prepare_data, forward_pass, MetricLogger
import torch
import torch.nn as nn
    

class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss, prt_fn):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()
    
def load_bn_stats(network, ref_model_path):
    """
    Load batch normalization statistics from a reference model to the current network.
    
    Args:
        network: The current network model where BN stats will be copied to
        ref_model_path: Path to the reference model checkpoint
    """
    # Check if the reference model path exists
    if not os.path.isfile(ref_model_path):
        raise FileNotFoundError(f"Reference model not found at: {ref_model_path}")
    
    # Load the reference model state dict
    ref_state_dict = torch.load(ref_model_path, map_location='cpu')
    
    # If the loaded object is a complete checkpoint (not just state_dict)
    if 'state_dict' in ref_state_dict:
        ref_state_dict = ref_state_dict['state_dict']
    
    # Get the current network state dict
    current_state_dict = network.state_dict()
    
    # Dictionary to store BN parameters and stats
    bn_params = {}
    
    # Find all BN layers' parameters in the reference model
    for key in ref_state_dict.keys():
        # Check if the parameter belongs to a BN layer
        if '.running_mean' in key or '.running_var' in key or '.weight' in key or '.bias' in key:
            if any(x in key for x in ['.bn', 'BatchNorm']):
                bn_params[key] = ref_state_dict[key]
    
    # Counter for BN layers
    bn_layers_updated = 0
    
    # Update the current network's BN layers with reference model's stats
    for key in bn_params:
        if key in current_state_dict:
            if current_state_dict[key].shape == bn_params[key].shape:
                current_state_dict[key].copy_(bn_params[key])
                bn_layers_updated += 1
            else:
                print(f"Shape mismatch for {key}, skipping: {current_state_dict[key].shape} vs {bn_params[key].shape}")
    
    # Load the updated state dict back to the network
    network.load_state_dict(current_state_dict, strict=False)
    
    print(f"Successfully updated {bn_layers_updated} BN parameters from {ref_model_path}")
    
    return network

def freeze_normalization_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # print(f"Freezing normalization layer: {name}")
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
class MyOldEvaluator:
    def __init__(self, eval_dataset, test_dataloader, network):
        self.eval_dataset = eval_dataset
        self.test_dataloader = test_dataloader
        self.network = copy.deepcopy(network)

    def evaluate(self, avgmodel = None, test_dataloader = None):
        if avgmodel is not None:
            self.network.load_state_dict(avgmodel.state_dict())
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        id_metric = self.network_evaluation()
        return id_metric
    def network_evaluation(self):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(self.test_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()
        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        return metric


class IIDMax(SWADBase):
    """SWAD start from iid max acc and select last by iid max swa acc"""

    def __init__(self, evaluator, **kwargs):
        self.iid_max_acc = 0.0
        self.swa_max_acc = 0.0
        self.avgmodel = None
        self.final_model = None
        self.evaluator = evaluator

    def update_and_evaluate(self, segment_swa, val_acc):
        if self.iid_max_acc < val_acc:
            self.iid_max_acc = val_acc
            self.avgmodel = swa_utils.AveragedModel(segment_swa.module, rm_optimizer=True)
            self.avgmodel.start_step = segment_swa.start_step

        self.avgmodel.update_parameters(segment_swa.module)
        self.avgmodel.end_step = segment_swa.end_step

        # evaluate
        swa_val_acc = self.evaluator.evaluate(self.avgmodel)
        if swa_val_acc > self.swa_max_acc:
            self.swa_max_acc = swa_val_acc
            self.final_model = copy.deepcopy(self.avgmodel)

    def get_final_model(self):
        return self.final_model


class LossValley(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, evaluator, n_converge, n_tolerance, tolerance_ratio, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """
        self.evaluator = evaluator
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)
        self.converge_Q_info = deque(maxlen=n_converge)
        self.smooth_Q_info = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    def update_and_evaluate(self, segment_swa, val_acc):
        if self.dead_valley:
            return

        frozen = copy.deepcopy(segment_swa.cpu())
        frozen.end_loss = 1-val_acc
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)
        self.converge_Q_info.append((frozen.start_step, frozen.end_step, frozen.end_loss))
        self.smooth_Q_info.append((frozen.start_step, frozen.end_step, frozen.end_loss))

        if not self.is_converged:
            if len(self.converge_Q) < self.n_converge:
                return

            min_idx = np.argmin([model.end_loss for model in self.converge_Q])
            untilmin_segment_swa = self.converge_Q[min_idx]  # until-min segment swa.
            if min_idx == 0:
                self.converge_step = self.converge_Q[0].end_step
                self.final_model = swa_utils.AveragedModel(untilmin_segment_swa)

                th_base = np.mean([model.end_loss for model in self.converge_Q])
                self.threshold = th_base * (1.0 + self.tolerance_ratio)

                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    start_idx = 0
                    for i in reversed(range(len(Q))):
                        model = Q[i]
                        if model.end_loss > self.threshold:
                            start_idx = i + 1
                            break
                    for model in Q[start_idx + 1 :]:
                        self.final_model.update_parameters(
                            model, start_step=model.start_step, end_step=model.end_step
                        )
                print(
                    f"Model converged at step {self.converge_step}, "
                    f"Start step = {self.final_model.start_step}; "
                    f"Threshold = {self.threshold:.6f}, "
                )
            return

        if self.smooth_Q[0].end_step < self.converge_step:
            return

        # converged -> loss valley
        min_vloss = self.get_smooth_loss(0)
        if min_vloss > self.threshold:
            self.dead_valley = True
            print(f"Valley is dead at step {self.final_model.end_step}")
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model, start_step=model.start_step, end_step=model.end_step
        )

    def get_final_model(self):
        if not self.is_converged:
            print(
                "Requested final model, but model is not yet converged; return last model instead"
            )
            return self.converge_Q[-1].cuda()

        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa, step=segment_swa.end_step)

        return self.final_model.cuda()

class SWADTrainer(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        # 初始化 SWAD 方法和相关参数
        super(SWADTrainer, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.swad_method = args.swad_method  # "IIDMax" 或 "LossValley"
        self.swad_args = {
            "evaluator": None,
            "n_converge": args.n_converge,
            "n_tolerance": args.n_tolerance,
            "tolerance_ratio": args.tolerance_ratio,
        }
        self.swad_eval_iter = args.swad_eval_iter

        if self.swad_method == "IIDMax":
            self.swad_handler = IIDMax(**self.swad_args)
        elif self.swad_method == "LossValley":
            self.swad_handler = LossValley(**self.swad_args)
        else:
            raise NotImplementedError(f"Unsupported SWAD method: {self.swad_method}")
    
    def get_evaluator(self):
        self.eval_dataset.mode = 1
        self.eval_dataset.update_current_timestamp(self.train_dataset.current_time)
        test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                            batch_size=self.mini_batch_size,
                                            num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        return MyOldEvaluator(self.eval_dataset, test_id_dataloader, self.network)

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        load_bn_stats(self.network, self.args.ref_bn_model_path)
        freeze_normalization_layers(self.network)
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        self.swad_handler.evaluator = self.get_evaluator()
        swad_algorithm = swa_utils.AveragedModel(self.network)
        for step, (x, y) in enumerate(dataloader):
            
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                               self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            swad_algorithm.update_parameters(self.network, step=step)

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            #-----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=(loss).item())
            if step % self.args.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "timestamp: {timestamp}",
                            f"[iter: {step}/{stop_iters}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        eta=eta_string,
                        timestamp=self.train_dataset.current_time,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )

            if step % (self.swad_eval_iter) == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                # pdb.set_trace()
                swa_acc = self.swad_handler.evaluator.evaluate(swad_algorithm.module, test_id_dataloader)
                self.swad_handler.update_and_evaluate(swad_algorithm, swa_acc)
                swad_algorithm = swa_utils.AveragedModel(self.network)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}, avg acc: {}".format(step, stop_iters, timestamp, self.eval_metric, acc * 100.0, swa_acc * 100.0))
                print("queues:", self.swad_handler.converge_Q_info, self.swad_handler.smooth_Q_info)
                freeze_normalization_layers(self.network)
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        final_model = self.swad_handler.get_final_model()
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                self.logger.info("Merged ID test {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
                id_avg_acc = self.swad_handler.evaluator.evaluate(final_model.module, test_id_dataloader)
                self.logger.info("SWAD Merged ID test {}: \t{:.3f}\n".format(self.eval_metric, id_avg_acc * 100.0))
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
                try:
                    ood_avg_acc = self.swad_handler.evaluator.evaluate(final_model.module, test_ood_dataloader)
                except:
                    pdb.set_trace()
                self.logger.info("SWAD OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, ood_avg_acc * 100.0))
                
        if len(metrics) >= 2:
            self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    # def evaluate_offline(self):
    #     # 执行基于 SWAD 的模型评估
    #     self.logger.info("\nEvaluating SWAD offline...\n")
    #     super().evaluate_offline()

    #     if self.swad_handler:
    #         self.logger.info("Finalizing SWAD model...")
    #         final_model = self.swad_handler.get_final_model()
    #         if final_model:
    #             # 如果需要，可以将最终的 SWAD 模型保存到文件
    #             torch.save(final_model.state_dict(), f"{self.args.log_dir}/swad_final_model.pth")
    #             self.logger.info("SWAD model saved successfully.")
    #         else:
    #             self.logger.warning("SWAD handler did not return a final model.")

    def evaluate_sw_model(self, eval_dataset):
        # 用于评估 SWAD 模型性能的辅助方法
        dataloader = FastDataLoader(
            dataset=eval_dataset,
            batch_size=self.mini_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate_fn,
        )
        self.network.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = prepare_data(x, y, str(eval_dataset))
                logits = self.network(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        self.network.train()
        freeze_normalization_layers(self.network)
        return accuracy, avg_loss
