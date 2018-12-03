# mario.ai

Reinforcement Learning in Super Mario Bros.

## References
- [`gym`](https://github.com/openai/gym), [@openai](https://github.com/openai) [[website]](https://openai.com)
- [`gym-super-mario`](https://github.com/ppaquette/gym-super-mario/), [@ppaquette](https://github.com/ppaquette)
- [`gym-super-mario-bros`](https://github.com/Kautenja/gym-super-mario-bros/), [@Kautenja](https://github.com/Kautenja/)
- [`pytorch-a3c`](https://github.com/ikostrikov/pytorch-a3c), [@ikostrikov](https://github.com/ikostrikov/)

![Latex](https://latex.codecogs.com/png.latex?%5Clarge%20%5CLaTeX) via CodeCogs ![EqnEditor](https://www.codecogs.com/latex/eqneditor.php)


## Asynchronous Advantage Actor Critic (A3C)

![equation](https://latex.codecogs.com/png.latex?%5Clarge%20L_%7Bi%7D%28%5CTheta_%7Bi%7D%29%20%3D%20%5Cmathbb%7BE%7D%5Br%20&plus;%20%5Cgamma%20%5Cmax_%7Ba%5E%7B%27%7D%7DQ%28s%5E%7B%27%7D%2C%20a%5E%7B%27%7D%3B%20%5CTheta_%7Bi-1%7D%29%20-%20Q%28s%2C%20a%3B%20Q_%7Bi%7D%29%5D%5E%7B2%7D)

![equation](https://latex.codecogs.com/png.latex?%5Clarge%20g%20%3D%20%5Calpha%20g%20&plus;%20%281%20-%20%5Calpha%29%5CDelta%5CTheta%5E%7B2%7D)

![equation](https://latex.codecogs.com/png.latex?%5Clarge%20%5CTheta_%7Bt%20&plus;%201%7D%20%5Cleftarrow%20%5CTheta%20-%20%5Ceta%20%5Cfrac%7B%5CDelta%20%5CTheta%7D%7B%5Csqrt%7Bg%20&plus;%20%5Cvarepsilon%7D%7D)
