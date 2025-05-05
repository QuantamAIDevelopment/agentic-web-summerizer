from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="agentic-web-summarizer",
    version="0.1.0",
    author="Agentic AI Developer",
    author_email="example@example.com",
    description="A tool that uses AI agents to summarize web content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentic-web-summarizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "web-summarizer=cli:main",
        ],
    },
)