from setuptools import setup, find_packages

setup(
    name="crystal_toolkit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],  # 依赖列表（如"requests>=2.25.1"）
    entry_points={"console_scripts": ["mycli=my_package.cli:main"]},  # 命令行工具配置
)
