Наблюдения:
    1) PolicySmoother
        наблюдалось улучшение значений при большем колличестве эпох
        Episode: 1
            Reward mean: -118.574
            Elite sessions (386)
        ...
        Episode: 10
            Reward mean: -84.1045
            Elite sessions (274)
        ...
        Episode: 20
            Reward mean: -54.3145
            Elite sessions (140)
        ...
        Episode: 30
            Reward mean: -46.546
            Elite sessions (428)
        ...
        Episode: 40
            Reward mean: -35.0735
            Elite sessions (428)
        ...
        Episode: 50
            Reward mean: -15.0035
            Elite sessions (405)
        ...

        замечание 1: работа сглаживания по policy требует
        большого колличества эпизодов так как при меньшем lambda
        значение старой policy больше, чем новой, то есть предыдущие
        вероятности не дают ухудшить результаты,
        но в то же время и не дают быстро "изучать" окружение
        рандомными решениями агента (как пример см. замечание 3)

        замечание 2: в данном случае агент показал результат: 23 / -11
        но предполагаю, что это из-за недостаточного кол-ва эпох,
        ведь тенденция на улучшение заметна.

        замечание 3: при параметре = 1
        работа без сглаживания как и в дз 1(1): 11 / 10


    2) LaplaceSmoother
        предоставляет возможность выбирать значение влияния действия на
        обучение, чем больше лямбда - тем больше обучение (те. рандомные действия)
        чем меньше, тем больше знания предыдущих ходов.

        замечание: для наилучший результатов необходимо
        увеличить кол-во эпизодов и сессий
        при этом уменьшить квантиль, чтобы брать по-больше сессий

        laplace_smother = LaplaceSmoother(2)
        total_epochs = 10
        total_sessions = 8000
        max_steps = 30
        q_param = 0.3

        Episode: 1
            Reward mean: -117.87325
            Elite sessions (4965)
        Episode: 2
            Reward mean: -103.9195
            Elite sessions (4504)
        Episode: 3
            Reward mean: -88.88775
            Elite sessions (5367)
        Episode: 4
            Reward mean: -78.123875
            Elite sessions (4345)
        Episode: 5
            Reward mean: -65.31925
            Elite sessions (5184)
        Episode: 6
            Reward mean: -56.895125
            Elite sessions (5186)
        Episode: 7
            Reward mean: -49.894125
            Elite sessions (4942)
        Episode: 8
            Reward mean: -43.633375
            Elite sessions (4439)
        Episode: 9
            Reward mean: -38.139375
            Elite sessions (3272)
        Episode: 10
            Reward mean: -33.820875
            Elite sessions (5393)

        счет: 30 / -30