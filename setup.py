from setuptools import setup, find_packages

setup(
    name="meta_bidding",
    version="0.0",
    author="Jinyu Liu",
    author_email="jinyu.thu@gmail.com",
    description="The environment for meta bidding accross multiple electricity markets",

    # 项目主页
    url="http://eilab.group/", 

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(
        exclude=["raw_data","raw_data.*","test","test.*"]
    )
)