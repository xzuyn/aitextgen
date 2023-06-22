from setuptools import setup

setup(
    name="aitextgen",
    packages=["aitextgen"],  # this must be the same as the name above
    version="0.6.1",
    description="A robust Python tool for text-based AI training and generation using GPT-2.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/aitextgen",
    keywords=["gpt-2", "gpt2", "text generation", "ai"],
    classifiers=[],
    license="MIT",
    entry_points={"console_scripts": ["aitextgen=aitextgen.cli:aitextgen_cli"]},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        absl-py==1.4.0,
        aiohttp==3.8.4,
        aiosignal==1.3.1,
        appdirs==1.4.4,
        async-timeout==4.0.2,
        attrs==23.1.0,
        cachetools==5.3.1,
        certifi==2023.5.7,
        charset-normalizer==3.1.0,
        click==8.1.3,
        colorama==0.4.6,
        docker-pycreds==0.4.0,
        filelock==3.12.2,
        fire==0.5.0,
        frozenlist==1.3.3,
        fsspec==2023.6.0,
        gitdb==4.0.10,
        GitPython==3.1.31,
        google-auth==2.20.0,
        google-auth-oauthlib==1.0.0,
        grpcio==1.54.2,
        h5py==3.8.0,
        huggingface-hub==0.15.1,
        idna==3.4,
        importlib-metadata==6.6.0,
        Keras-Applications==1.0.8,
        Keras-Preprocessing==1.1.2,
        lightning-utilities==0.8.0,
        Markdown==3.4.3,
        MarkupSafe==2.1.3,
        multidict==6.0.4,
        numpy==1.24.3,
        oauthlib==3.2.2,
        packaging==23.1,
        pathtools==0.1.2,
        protobuf==4.23.3,
        psutil==5.9.5,
        pyasn1==0.5.0,
        pyasn1-modules==0.3.0,
        pytorch-lightning==2.0.3,
        PyYAML==6.0,
        regex==2023.6.3,
        requests==2.31.0,
        requests-oauthlib==1.3.1,
        rsa==4.9,
        scipy==1.10.1,
        sentry-sdk==1.25.1,
        setproctitle==1.3.2,
        six==1.16.0,
        smmap==5.0.0,
        tensorboard==2.13.0,
        tensorboard-data-server==0.7.1,
        termcolor==2.3.0,
        tokenizers==0.13.3,
        torch==1.13.1,
        torchmetrics==0.11.4,
        tqdm==4.65.0,
        transformers==4.26.1,
        typing-extensions==4.6.3,
        urllib3==1.26.16,
        wandb==0.15.4,
        Werkzeug==2.3.6,
        yarl==1.9.2,
        zipp==3.15.0,
    ],
)
