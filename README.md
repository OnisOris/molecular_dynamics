# 2D Molecular Dynamics of Potassium

Учебный проект по заданию `Задание.docx`. Модель строго двумерная: у K есть
только координаты `(x, y)` и скорости `(vx, vy)`. Внутри ядра используются
только SI-единицы: m, kg, s, K, J на частицу. Давление двумерной модели имеет
единицы `J/m^2 = N/m`; эквивалентное 3D-давление получается делением на
толщину слоя `z`.

## Фиксированный вариант

```text
element       K
concentration 4 mol/L
temperature   900 C = 1173.15 K
box           24 nm x 24 nm
particles     1035
sigma_K       0.4250 nm
epsilon_K/kB  850 K
cutoff        2.5 sigma
dt            1 fs
```

Число частиц не является произвольным. Для `Vref = 1 L`:

```text
Nref = C Vref NA = 2.408856304e24
V1   = Vref / Nref
z    = cubert(V1) = 0.745984 nm
N2D  = Nref z / cubert(Vref)
Nbox = N2D (24 nm)^2 / (10 cm)^2 = 1035.054
```

В коде используется `round`, поэтому `N=1035`.

## Параметры и источники

- Масса K `39.0983 u`, или `6.49242546e-26 kg`: IUPAC/CIAAW atomic-weight
  data, также доступно через [PubChem potassium](https://pubchem.ncbi.nlm.nih.gov/element/Potassium).
- Для K-K используется фиксированная LJ 12-6 параметризация из проверенной
  литературной таблицы, принятой для этого варианта: `sigma=4.250 A` и
  `epsilon/kB=850 K`. Температурно-зависимая параметризация Eslami не
  используется, потому что тогда при temperature sweep менялся бы сам
  Hamiltonian.
- Для diamond carbon используется UFF `C_3` из Rappe et al., *UFF, a full
  periodic table force field for molecular mechanics and molecular dynamics
  simulations*, JACS 114 (1992), DOI
  [10.1021/ja00051a040](https://doi.org/10.1021/ja00051a040): `x=3.851 A`,
  `D=0.105 kcal/mol`. Так как UFF `x` является положением минимума, для
  обычного LJ 12-6 в коде используется `sigma=x/2^(1/6)`.
- Diamond (100) wall spacing: `a/sqrt(2)=0.2522 nm`, где
  `a=0.3567 nm`; reference geometry: [Argonne lattice constants](https://7id.xray.aps.anl.gov/calculators/crystal_lattice_parameters.html).

Из K и C-параметров вычисляются Lorentz-Berthelot K-C parameters:

```text
sigma_KC   = (sigma_K + sigma_C)/2 = 0.384043 nm
epsilon_KC = sqrt(epsilon_K epsilon_C) = 1.7620 kJ/mol
```

## Математическая модель

### LJ и LJTS

```text
U_LJ(r) = 4 epsilon ((sigma/r)^12 - (sigma/r)^6)
U_LJTS(r) = U_LJ(r) - U_LJ(rc), r < rc
U_LJTS(r) = 0,                         r >= rc
```

Energy shift делает потенциал непрерывным в cutoff, но сила в `rc` вообще не
обязана быть непрерывной. Force-shifted LJ намеренно не используется.

При `r_ij = r_i-r_j` сила на `i` вычисляется как `-grad_i U`:

```text
F_ij = 24 epsilon (2 sigma^12/r^14 - sigma^6/r^8) r_ij
```

Пары вычисляются один раз, поэтому `F_ji=-F_ij`.

### Minimum image и границы

Для периодической оси:

```text
dr_axis -= L_axis round(dr_axis / L_axis)
```

Task 1 использует PBC по `x` и `y`. Task 3 использует четыре фиксированные
линии C-атомов. Task 4 использует fixed walls по `x` и PBC по `y`; minimum
image в Task 4 применяется только по `y`.

### Инициализация и интегрирование

Каждая компонента скорости имеет распределение `N(0, kB*T/m)`. После этого
вычитается скорость центра масс и выполняется один rescale к целевой
температуре. При удаленных двух COM-степенях свободы:

```text
T = 2 K / ((2N-2) kB)
```

Velocity Verlet:

```text
r_new = r + v dt + 0.5 a dt^2
v_new = v + 0.5 (a_old + a_new) dt
```

### Энергия, температура и давление

```text
K = 0.5 m sum(v_i^2)
E = K + U
```

Для периодической системы используется двумерный virial:

```text
P2D = (K + 0.5 W) / A
W   = sum_{i<j} r_ij dot F_ij
```

Для идеального газа:

```text
P2D_ideal = N kB T / A = 0.0291041 N/m
P3D_equiv = P2D / z = 3.9014e7 Pa
```

Для Task 4 основной observable давления является механическое давление на
фиксированные x-стенки:

```text
P_left  = abs(sum(F_wall_left,x)) / Ly
P_right = abs(sum(F_wall_right,x)) / Ly
P_wall  = (P_left + P_right) / 2
```

## Фазовый критерий

Температура перехода не зашита. На каждом production snapshot строятся
connected components: две K-сущности связаны при `r < 1.5 sigma_K`. Основной
order parameter:

```text
M = size(largest cluster) / N
chi_M = N (mean(M^2) - mean(M)^2)
```

Температура crossover определяется максимумом `chi_M`; seed-to-seed range
используется как неопределенность. RDF и потенциальная энергия являются
дополнительными diagnostics. Оценка относится только к конечной 2D LJTS
модели со стенками, а не к реальной температуре кипения калия.

## UQ

`src.uq.run_simulation(T, N, config)` является внешней оберткой над тем же
MD solver. DOE использует `T={0.9T0,T0,1.1T0}`, `N={0.95N0,N0,1.05N0}` и
несколько seeds. Подгоняется:

```text
P(T,N) = a + b T + c N
dP/dT = b
dP/dN = c
```

Неопределенность входов распространяется численно sampling из независимых
uniform distributions. Это не смешивается с sensitivity: `b` и `c` описывают
наклон surrogate, а распределение `P` описывает variation output при заданных
диапазонах T и N.

## Запуск

Создать окружение и установить зависимости:

```bash
uv sync
```

Запуски из корня проекта:

```bash
uv run python -m experiments.task1_periodic
uv run python -m experiments.task2_equilibrium
uv run python -m experiments.task3_walls
uv run python -m experiments.task4_phase_transition
uv run python -m experiments.task5_uq
```

Для короткой проверки можно передать `--steps`, `--equilibration-steps` и
`--production-steps`. Полные sweep и equilibrium runs требуют заметного
времени, потому что параметры не подгоняются под ожидаемый ответ.

## Тесты и производительность

```bash
uv run pytest -q
```

Brute-force backend `O(N^2)` является эталоном. Cell-list backend используется
для production после сравнения force, energy и virial с brute-force. Тесты
проверяют LJ, LJTS cutoff, знак силы, Newton's third law, minimum image, PBC,
zero COM momentum, initial temperature, reproducibility, NVE energy drift,
cell-list equality и fixed-wall behavior.

Все графики, CSV и GIF сохраняются в `output/`.
