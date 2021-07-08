# Baechi: Fast Device Placement on Machine Learning Graphs (SoCC 2020)

## Install dependencies
* Install dependencies with Anaconda
```
$ conda install -y python=3.6 numpy=1.16 tensorflow-gpu=1.12 bazel=0.20.0 \
      networkx future matplotlib cvxopt scikit-learn
```
* Mosek
```
$ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek==8.1.82
```

Our code runs [MOSEK](https://www.mosek.com/) as an LP solver for SCT.
MOSEK provides a free personal academic license.
You can request a license at https://www.mosek.com/products/academic-licenses.
The license file (`mosek.lic`) should be placed at `$HOME/mosek`.

## Example usage
This example generates the placement of 4-layer GNMT v2 with a batch size of 128, a maximum sequence length of 40, and a vocabulary size of 30000.

* Build a Python program to place operators of an ML model.
```
$ bazel build :train
```

* Generate profiles.

```
$ ./bazel-bin/train \
    --costgen \
    --cost_path=/tmp/cost.pkl \
    --optimizer=adam \
    --batch_size=128 \
    --model_name=gnmt_v2 \
    --vocab_size=30000 \
    --max_seq_length=40 \
    --rnn_unit_type=lstm \
    --rnn_units=512 \
    --num_layers=4 \
    --encoder_type=gnmt \
    --num_gpus=4 \
    --residual \
    --colocate_grads_with_ops \
    --only_forward
```
This generates profiles of the forward pass and stores them at `/tmp/cost.pkl`.

* Generate a communication cost function between GPUs through the linear regression.

```
$ bazel build //utils:communication_benchmark
$ ./bazel-bin/utils/communication_benchmark
```

This runs a benchmark that transfers tensors between different GPUs for various tensor sizes.
By default, the benchmark transfers tensors from `GPU:0` to `GPU:1` with tensor sizes in the range [2<sup>0</sup>, 2<sup>29</sup>].
After the benchmark finishes, it prints out a generated communication cost function that
should be given as the `--comm_cost_coeffs` argument value for the placement.

An example output would be the following.
```
...
Communication cost function: 0.0001754 x + 134
```

* Place operators of GNMT v2 and measure average step times.

```
$ ./bazel-bin/train \
    --cost_path=/tmp/cost.pkl \
    --optimizer=adam \
    --batch_size=128 \
    --model_name=gnmt_v2 \
    --vocab_size=30000 \
    --max_seq_length=40 \
    --rnn_unit_type=lstm \
    --rnn_units=512 \
    --num_layers=4 \
    --encoder_type=gnmt \
    --num_gpus=4 \
    --residual \
    --colocate_grads_with_ops \
    --only_forward \
    --placement_method=m_etf \
    --placer_type=fusion \
    --grouper=coplace \
    --comm_cost_coeffs=0.0001754,134 \
    --memory_fraction=1.0
```

This runs the placement of GNMT v2 operators using m-ETF based on the forward operators.
When the placement is done, this measures the average step time of the placement results and prints it out.

## Docker image

A Docker image with all dependencies installed is available.

```
$ docker pull beomyeol/baechi
$ docker run -it --rm --gpus all beomyeol/baechi /bin/bash
```

This gives you direct access to the container with all GPUs enabled.
You can follow the example usage within the container.

## License
University of Illinois/NCSA Open Source License
