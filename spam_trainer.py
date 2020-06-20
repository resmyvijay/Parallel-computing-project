{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "spam-trainer.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "_rM-JcSlM27w"
      ],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/resmyvijay/Datamining/blob/master/spam_trainer.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "colab_type": "text",
        "id": "hLhZWxXQLAsx"
      },
      "source": [
        "## Prerequisites"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mxduhMI5J54f",
        "colab_type": "text"
      },
      "source": [
        "First, we load PySyft and PyTorch:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MY607kecrnH1",
        "colab_type": "code",
        "outputId": "6b6e72ff-111f-44ea-9aae-afbce575b5cd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        }
      },
      "source": [
        "!pip install syft torch==1.3.0 torchvision==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Looking in links: https://download.pytorch.org/whl/torch_stable.html\n",
            "Requirement already satisfied: syft in /usr/local/lib/python3.6/dist-packages (0.2.0a2)\n",
            "Requirement already satisfied: torch==1.3.0 in /usr/local/lib/python3.6/dist-packages (1.3.0+cu92)\n",
            "Requirement already satisfied: torchvision==0.4.1 in /usr/local/lib/python3.6/dist-packages (0.4.1+cu92)\n",
            "Requirement already satisfied: flask-socketio>=3.3.2 in /usr/local/lib/python3.6/dist-packages (from syft) (4.2.1)\n",
            "Requirement already satisfied: lz4>=2.1.6 in /usr/local/lib/python3.6/dist-packages (from syft) (2.2.1)\n",
            "Requirement already satisfied: websockets>=7.0 in /usr/local/lib/python3.6/dist-packages (from syft) (8.1)\n",
            "Requirement already satisfied: msgpack>=0.6.1 in /usr/local/lib/python3.6/dist-packages (from syft) (0.6.2)\n",
            "Requirement already satisfied: websocket-client>=0.56.0 in /usr/local/lib/python3.6/dist-packages (from syft) (0.56.0)\n",
            "Requirement already satisfied: zstd>=1.4.0.0 in /usr/local/lib/python3.6/dist-packages (from syft) (1.4.4.0)\n",
            "Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.6/dist-packages (from syft) (1.17.4)\n",
            "Requirement already satisfied: Flask>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from syft) (1.1.1)\n",
            "Requirement already satisfied: tblib>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from syft) (1.6.0)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision==0.4.1) (1.12.0)\n",
            "Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==0.4.1) (4.3.0)\n",
            "Requirement already satisfied: python-socketio>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from flask-socketio>=3.3.2->syft) (4.4.0)\n",
            "Requirement already satisfied: Werkzeug>=0.15 in /usr/local/lib/python3.6/dist-packages (from Flask>=1.0.2->syft) (0.16.0)\n",
            "Requirement already satisfied: Jinja2>=2.10.1 in /usr/local/lib/python3.6/dist-packages (from Flask>=1.0.2->syft) (2.10.3)\n",
            "Requirement already satisfied: click>=5.1 in /usr/local/lib/python3.6/dist-packages (from Flask>=1.0.2->syft) (7.0)\n",
            "Requirement already satisfied: itsdangerous>=0.24 in /usr/local/lib/python3.6/dist-packages (from Flask>=1.0.2->syft) (1.1.0)\n",
            "Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from pillow>=4.1.1->torchvision==0.4.1) (0.46)\n",
            "Requirement already satisfied: python-engineio>=3.9.0 in /usr/local/lib/python3.6/dist-packages (from python-socketio>=4.3.0->flask-socketio>=3.3.2->syft) (3.11.1)\n",
            "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from Jinja2>=2.10.1->Flask>=1.0.2->syft) (1.1.1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kehVN2WpJ9vF",
        "colab_type": "text"
      },
      "source": [
        "Because PyTorch does not yet support TensorFlow 2, we must limit the TensorFlow backend used by Google Colab to version 1.x as follows:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a_ilG7JYKQu5",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%tensorflow_version 1.x"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gdu0C973KDDR",
        "colab_type": "text"
      },
      "source": [
        "We then proceed to check for CUDA support that will be used by PyTorch and assign it to a variable for reference later:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_opY0kpwRVlb",
        "colab_type": "code",
        "outputId": "6709bec9-c44e-4f13-f78c-e82e74c9b182",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "import torch as tc\n",
        "\n",
        "if tc.cuda.is_available():\n",
        "  print(tc.cuda.get_device_name(0))\n",
        "  print(tc.cuda.device_count())\n",
        "  tc.set_default_tensor_type(tc.cuda.FloatTensor)\n",
        "  cuda_device = tc.device('cuda:0')"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Tesla P100-PCIE-16GB\n",
            "1\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ztiJdl33KHhf",
        "colab_type": "text"
      },
      "source": [
        "We import the PySyft library at this stage and assign the PyTorch tensor that we imported into the notebook earlier. This allows PySft to interface with PyTorch as it’s backend. We also create 2 worker nodes (Worker1 and Worker2) that will be used for our training later. To ease setup, we will create these workers as virtual nodes that are part of the same python runtime:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab_type": "code",
        "id": "ku19mUk4NJFd",
        "outputId": "54b9536c-6523-42f6-cbd6-e96da75c53f3",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "import syft as sf\n",
        "\n",
        "hook = sf.TorchHook(tc)\n",
        "w1 = sf.VirtualWorker(hook, id=\"worker1\")\n",
        "w2 = sf.VirtualWorker(hook, id=\"worker2\")"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:root:Torch was already hooked... skipping hooking process\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_rM-JcSlM27w",
        "colab_type": "text"
      },
      "source": [
        "## Worker Sanity Check"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JgkGLQ6gKZx8",
        "colab_type": "text"
      },
      "source": [
        "This section is to test whether the nodes are working and could be used for computation."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "E9XxS99UvuWM",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "x = tc.Tensor([3,2,1]).send(w1)\n",
        "print(x)\n",
        "print(x.location)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3rBSGjcQv4Jx",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "y = tc.Tensor([1,2,3]).send(w1)\n",
        "sum = x + y\n",
        "print(sum)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wP40qMqqwC4H",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "sum = sum.get()\n",
        "print(sum)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MOq_bqgnNph9",
        "colab_type": "text"
      },
      "source": [
        "## Actual Trainer"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tiVKBoh_LQOF",
        "colab_type": "text"
      },
      "source": [
        "We upload the msgtypes and msgtext pre-processed data into the notebook to be used for training:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab_type": "code",
        "outputId": "97b52a8d-4a2a-42e0-e671-03ed759fd4ad",
        "id": "6djwfkXrHNNx",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 108
        }
      },
      "source": [
        "from google.colab import files\n",
        "\n",
        "upl = files.upload()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-4b45e5ac-f19a-43b4-904f-09592977bc45\" name=\"files[]\" multiple disabled />\n",
              "     <output id=\"result-4b45e5ac-f19a-43b4-904f-09592977bc45\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving msgtexts.npy to msgtexts (1).npy\n",
            "Saving msgtypes.npy to msgtypes (1).npy\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AuVhpeQXLa0i",
        "colab_type": "text"
      },
      "source": [
        "After successful loading of the data into the notebook, we import it into the following variables, texts for the text messages and types for the message types:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V2IkY6SSt9vV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import io\n",
        "import numpy as np\n",
        "\n",
        "# Loading data\n",
        "texts = np.load(io.BytesIO(upl[\"msgtexts.npy\"]))\n",
        "texts = tc.tensor(texts).to(cuda_device, tc.long)\n",
        "types = np.load(io.BytesIO(upl[\"msgtypes.npy\"]))\n",
        "types = tc.tensor(types).to(cuda_device, tc.long)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zxMTwu62LoxV",
        "colab_type": "text"
      },
      "source": [
        "We split the data into 80% for training and 20% for testing, then spit it into 2 halves for Worker1 and Worker2:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7pq1h6E50PAA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Split training and test data\n",
        "test_pct = 0.2\n",
        "train_types = types[:-int(len(types)*test_pct)]\n",
        "train_texts = texts[:-int(len(types)*test_pct)]\n",
        "test_types = types[-int(len(types)*test_pct):]\n",
        "test_texts = texts[-int(len(types)*test_pct):]\n",
        "\n",
        "# Dataset split (one half for w1, other half for w2)\n",
        "train_idx = int(len(train_types)/2)\n",
        "test_idx = int(len(test_types)/2)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aIs6N5mnLyAO",
        "colab_type": "text"
      },
      "source": [
        "We then send the data sets (both test and training datasets) to the workers for processing. Note that the data can’t be processed in the notebook directly, as that would not be federated learning. It has to be done on separate (worker) nodes and not visible to the master (notebook):"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "u2AUqOtF0_QR",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "w1_train_dataset = sf.BaseDataset(train_texts[:train_idx], \n",
        "                                  train_types[:train_idx]).send(w1)\n",
        "w2_train_dataset = sf.BaseDataset(train_texts[train_idx:], \n",
        "                                    train_types[train_idx:]).send(w2)\n",
        "w1_test_dataset = sf.BaseDataset(test_texts[:test_idx], \n",
        "                                  test_types[:test_idx]).send(w1)\n",
        "w2_test_dataset = sf.BaseDataset(test_texts[test_idx:], \n",
        "                                  test_types[test_idx:]).send(w2)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Kh-KQDj1MZGc",
        "colab_type": "text"
      },
      "source": [
        "We define a global value for the batch processing size that will be performed by each worker:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NIditt29McSF",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "BATCH_SIZE = 32"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MXKHbjAFMsQn",
        "colab_type": "text"
      },
      "source": [
        "We specify that the datasets that were sent to the workers are to be processed in a federated manner:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WZ9BkXyn2gv-",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "federated_train_dataset = sf.FederatedDataset([w1_train_dataset, w2_train_dataset])\n",
        "federated_test_dataset = sf.FederatedDataset([w1_test_dataset, w2_test_dataset])\n",
        "\n",
        "federated_train_loader = sf.FederatedDataLoader(federated_train_dataset, \n",
        "                                                shuffle=True, batch_size=BATCH_SIZE)\n",
        "federated_test_loader = sf.FederatedDataLoader(federated_test_dataset, \n",
        "                                               shuffle=False, batch_size=BATCH_SIZE)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QbassBzJNJMu",
        "colab_type": "text"
      },
      "source": [
        "This is th GRU RNN model that will used for the training:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mh-LuFgR6e6u",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from torch import nn\n",
        "import torch as tc\n",
        "\n",
        "class GRUCell(nn.Module):\n",
        "\n",
        "    def __init__(self, input_size, hidden_size, bias=True):\n",
        "        super(GRUCell, self).__init__()\n",
        "        self.input_size = input_size\n",
        "        self.hidden_size = hidden_size\n",
        "        self.bias = bias\n",
        "\n",
        "        # reset gate\n",
        "        self.fc_ir = nn.Linear(input_size, hidden_size, bias=bias)\n",
        "        self.fc_hr = nn.Linear(hidden_size, hidden_size, bias=bias)\n",
        "\n",
        "        # update gate\n",
        "        self.fc_iz = nn.Linear(input_size, hidden_size, bias=bias)\n",
        "        self.fc_hz = nn.Linear(hidden_size, hidden_size, bias=bias)\n",
        "\n",
        "        # new gate\n",
        "        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)\n",
        "        self.fc_hn = nn.Linear(hidden_size, hidden_size, bias=bias)\n",
        "\n",
        "        self.init_parameters()\n",
        "\n",
        "    def init_parameters(self):\n",
        "        std = 1.0 / np.sqrt(self.hidden_size)\n",
        "        for w in self.parameters():\n",
        "            w.data.uniform_(-std, std)\n",
        "\n",
        "    def forward(self, x, h):\n",
        "\n",
        "        x = x.view(-1, x.shape[1])\n",
        "\n",
        "        i_r = self.fc_ir(x)\n",
        "        h_r = self.fc_hr(h)\n",
        "        i_z = self.fc_iz(x)\n",
        "        h_z = self.fc_hz(h)\n",
        "        i_n = self.fc_in(x)\n",
        "        h_n = self.fc_hn(h)\n",
        "\n",
        "        resetgate = tc.sigmoid(i_r + h_r)\n",
        "        inputgate = tc.sigmoid(i_z + h_z)\n",
        "        newgate = tc.tanh(i_n + (resetgate * h_n))\n",
        "\n",
        "        hy = newgate + inputgate * (h - newgate)\n",
        "\n",
        "        return hy\n",
        "\n",
        "\n",
        "class GRU(nn.Module):\n",
        "    def __init__(self, vocab_size, output_size=1, embedding_dim=50, hidden_dim=10, bias=True, dropout=0.2):\n",
        "        super(GRU, self).__init__()\n",
        "\n",
        "        self.hidden_dim = hidden_dim\n",
        "        self.output_size = output_size\n",
        "\n",
        "        # Dropout layer\n",
        "        self.dropout = nn.Dropout(p=dropout)\n",
        "        # Embedding layer\n",
        "        self.embedding = nn.Embedding(vocab_size, embedding_dim)\n",
        "        # GRU Cell\n",
        "        self.gru_cell = GRUCell(embedding_dim, hidden_dim)\n",
        "        # Fully-connected layer\n",
        "        self.fc = nn.Linear(hidden_dim, output_size)\n",
        "        # Sigmoid layer\n",
        "        self.sigmoid = nn.Sigmoid()\n",
        "\n",
        "    def forward(self, x, h):\n",
        "\n",
        "        batch_size = x.shape[0]\n",
        "\n",
        "        # Deal with cases were the current batch_size is different from general batch_size\n",
        "        # It occurrs at the end of iteration with the Dataloaders\n",
        "        if h.shape[0] != batch_size:\n",
        "            h = h[:batch_size, :].contiguous()\n",
        "\n",
        "        # Apply embedding\n",
        "        x = self.embedding(x)\n",
        "\n",
        "        # GRU cells\n",
        "        for t in range(x.shape[1]):\n",
        "            h = self.gru_cell(x[:,t,:], h)\n",
        "\n",
        "        # Output corresponds to the last hidden state\n",
        "        out = h.contiguous().view(-1, self.hidden_dim)\n",
        "\n",
        "        # Dropout and fully-connected layers\n",
        "        out = self.dropout(out)\n",
        "        sig_out = self.sigmoid(self.fc(out))\n",
        "\n",
        "        return sig_out, h"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eGSzxxtbNjJm",
        "colab_type": "text"
      },
      "source": [
        "We define the number of training epochs that we want to perform using the GRU model we have defined earlier:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6ZErgqBE6lJE",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "EPOCHS = 15\n",
        "CLIP = 5\n",
        "LR = 0.1\n",
        "\n",
        "# Model parameters\n",
        "VOCAB_SIZE = int(texts.max()) + 1\n",
        "EMBEDDING_DIM = 50\n",
        "HIDDEN_DIM = 10\n",
        "DROPOUT = 0.2\n",
        "\n",
        "# Initiating the model\n",
        "model = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s4F3UTGSN3ff",
        "colab_type": "text"
      },
      "source": [
        "We perform the training and test of the model defined earlier:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FPd7SDAV7NnA",
        "colab_type": "code",
        "outputId": "0d6662af-2c36-4b91-d70c-b0b1ccbc8169",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 272
        }
      },
      "source": [
        "from sklearn.metrics import roc_auc_score\n",
        "\n",
        "# Defining the loss and optimizer\n",
        "criterion = nn.BCELoss()\n",
        "optimizer = tc.optim.SGD(model.parameters(), lr=LR)\n",
        "\n",
        "for e in range(EPOCHS):\n",
        "  # To track the amount of loss\n",
        "  losses = []\n",
        "\n",
        "  # Batch processing loop for training\n",
        "  for texts, types in federated_train_loader:\n",
        "      # Location of current batch\n",
        "      worker = texts.location\n",
        "      # Initialize hidden state and send it to the worker\n",
        "      h = tc.Tensor(tc.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)\n",
        "      # Send the model to the current worker\n",
        "      model.send(worker)\n",
        "      # Accumulated gradients to zero before optimization step\n",
        "      optimizer.zero_grad()\n",
        "      # Output from the model\n",
        "      output, _ = model(texts, h)\n",
        "      # Calculate the loss and perform backprop\n",
        "      loss = criterion(output.squeeze(), types.float())\n",
        "      loss.backward()\n",
        "      # Clipping the gradient to avoid explosion\n",
        "      nn.utils.clip_grad_norm_(model.parameters(), CLIP)\n",
        "      # Backpropagation step\n",
        "      optimizer.step() \n",
        "      # Get the model back to the master\n",
        "      model.get()\n",
        "      losses.append(loss.get())\n",
        "\n",
        "  # Evaluate the model\n",
        "  model.eval()\n",
        "\n",
        "  with tc.no_grad():\n",
        "    test_preds = []\n",
        "    test_types_list = []\n",
        "    eval_losses = []\n",
        "\n",
        "    for texts, types in federated_test_loader:\n",
        "      # Location of current batch\n",
        "      worker = texts.location\n",
        "      # Initialize hidden state and send it to worker\n",
        "      h = tc.Tensor(tc.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)    \n",
        "      # Send the model to the worker\n",
        "      model.send(worker)\n",
        "      output, _ = model(texts, h)\n",
        "      loss = criterion(output.squeeze(), types.float())\n",
        "      eval_losses.append(loss.get())\n",
        "      preds = output.squeeze().get()\n",
        "      test_preds += list(preds.cpu().numpy())\n",
        "      test_types_list += list(types.get().cpu().numpy().astype(int))\n",
        "      # Get the model back to the master\n",
        "      model.get()\n",
        "            \n",
        "    score = roc_auc_score(test_types_list, test_preds)\n",
        "    \n",
        "  print(\"Epoch {}/{}...  \\    AUC: {:.3%}...  \\    Training loss: {:.5f}...  \\\n",
        "  Validation loss: {:.5f}\".format(e+1, EPOCHS, score, sum(losses)/len(losses), sum(eval_losses)/len(eval_losses)))\n",
        "    \n",
        "  model.train()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/15...  \\    AUC: 75.826%...  \\    Training loss: 0.41254...    Validation loss: 0.35196\n",
            "Epoch 2/15...  \\    AUC: 81.758%...  \\    Training loss: 0.34992...    Validation loss: 0.31816\n",
            "Epoch 3/15...  \\    AUC: 87.728%...  \\    Training loss: 0.30965...    Validation loss: 0.27454\n",
            "Epoch 4/15...  \\    AUC: 93.135%...  \\    Training loss: 0.25504...    Validation loss: 0.21443\n",
            "Epoch 5/15...  \\    AUC: 96.853%...  \\    Training loss: 0.19906...    Validation loss: 0.14857\n",
            "Epoch 6/15...  \\    AUC: 97.855%...  \\    Training loss: 0.14752...    Validation loss: 0.12568\n",
            "Epoch 7/15...  \\    AUC: 97.994%...  \\    Training loss: 0.11948...    Validation loss: 0.11120\n",
            "Epoch 8/15...  \\    AUC: 98.189%...  \\    Training loss: 0.10143...    Validation loss: 0.10220\n",
            "Epoch 9/15...  \\    AUC: 98.448%...  \\    Training loss: 0.08820...    Validation loss: 0.10390\n",
            "Epoch 10/15...  \\    AUC: 98.374%...  \\    Training loss: 0.07665...    Validation loss: 0.10007\n",
            "Epoch 11/15...  \\    AUC: 98.303%...  \\    Training loss: 0.07172...    Validation loss: 0.11391\n",
            "Epoch 12/15...  \\    AUC: 98.194%...  \\    Training loss: 0.06281...    Validation loss: 0.09796\n",
            "Epoch 13/15...  \\    AUC: 98.048%...  \\    Training loss: 0.05852...    Validation loss: 0.11699\n",
            "Epoch 14/15...  \\    AUC: 98.313%...  \\    Training loss: 0.05409...    Validation loss: 0.09376\n",
            "Epoch 15/15...  \\    AUC: 98.159%...  \\    Training loss: 0.04683...    Validation loss: 0.09330\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}
