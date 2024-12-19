from setuptools import find_packages, setup

setup(
    name="codenames-ai-client",
    version="0.1",
    packages=find_packages(),
    install_requires=["gensim", "nltk", "requests", "python-socketio[client]"],
)
