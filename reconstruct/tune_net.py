"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import hyperparams
from utils import io_utils, misc
from reconstruct.base_reconstructor import BaseReconstructor

import torch
from tqdm import tqdm
from time import time


class Tuner(BaseReconstructor):
    def __init__(self, device, generator, debug_out_path=None, l2_weight=hyperparams.l2_weight):
        super().__init__(device, generator, hyperparams.tune_steps, debug_out_path, l2_weight)

    def set_optimization(self):
        self.generator.train()
        for p in self.generator.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=hyperparams.tune_lr)
    """
    def reconstruct(self, dataset):
        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            # TODO(1): batched training
            for sample in dataset:
                anchor = sample.w_code.cuda()
                target = sample.img.cuda()

                synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                loss = self.reconstruction_loss(synth, target)

                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()

            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')

        return self.generator
    """

    def reconstruct(self, dataloader):
        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)
            start_time = time()

            for batch in dataloader:
                
                anchors = batch['w_code'].to(self.device)
                targets = batch['img'].to(self.device)
                names = batch['name']
                loss = 0.0

                for anchor, target, name in zip(anchors, targets, names):
                    # accumulate gradients
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                    loss += self.reconstruction_loss(synth, target)

                    if to_visualize:
                        io_utils.save_images(synth, self.debug_out_path.joinpath(name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()

            end_time = time()
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}. Time: {end_time - start_time:.2f}s')

        return self.generator

    def reconstruct_with_replay(self, dataloader, replay_iterator):
        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)
            start_time = time()

            for batch in dataloader:
                anchors = batch['w_code'].to(self.device)
                targets = batch['img'].to(self.device)
                names = batch['name']

                replay_batch = next(replay_iterator)
                replay_anchors = replay_batch['w_code'].to(self.device)
                replay_targets = replay_batch['img'].to(self.device)
                replay_names = replay_batch['name']

                loss = 0.0
                for anchor, target, name, replay_anchor, replay_target, replay_name in zip(anchors, targets, names, replay_anchors, replay_targets, replay_names):
                    # accumulate gradients
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                    loss += self.reconstruction_loss(synth, target)

                    synth_replay = self.generator(replay_anchor, noise_mode='const', force_fp32=True)
                    loss += self.reconstruction_loss(synth_replay, replay_target)

                    if to_visualize:
                        io_utils.save_images(synth, self.debug_out_path.joinpath(name))
                        io_utils.save_images(synth_replay, self.debug_out_path.joinpath(replay_name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()

            end_time = time()
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}. Time: {end_time - start_time:.2f}s')

        return self.generator