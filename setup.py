from setuptools import setup
from pathlib import Path

# Automatically discover all top-level modules in src/ directory
py_files = [p.stem for p in Path(__file__).parent.joinpath("src").glob("*.py")]

setup(
    name="heat_sdc_solver",  # Cambia este nombre según tu preferencia
    version="0.1.0",  # Versión inicial
    description="Solver SDC para la ecuación de calor usando Firedrake",
    author="Omar Khalil Abuali",  # Ajusta tu nombre o mantenlo así
    author_email="omar.khalil@example.com",  # Ajusta tu correo
    license="MIT",  # O la licencia que uses
    python_requires=">=3.8",  # Firedrake requiere Python 3.8+ típicamente
    install_requires=[  # Dependencias
        "firedrake",
        "pandas>=1.0",
        "numpy>=1.18",
        "matplotlib>=3.0",
    ],
    py_modules=py_files,  # Módulos individuales en src/
    package_dir={"": "src"},  # Raíz de los módulos
    entry_points={  # Script de consola para lanzar la simulación
        "console_scripts": [
            "run-heat=run_heat:main",
        ],
    },
)
