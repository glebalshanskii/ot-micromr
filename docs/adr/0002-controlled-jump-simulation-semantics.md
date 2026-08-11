# ADR-0002: Controlled numerical semantics jump simulator

- **Статус:** Accepted
- **Дата:** 2026-08-11
- **Связанный этап:** P1; исполняется в P3
- **Связанный план:** [`docs/plan.md`](../plan.md)

## Контекст

Целевой объект статьи задан не дискретной цепью, а непрерывновременным
jump-diffusion. Efficient price удовлетворяет

$$
dX_t=\sigma_X\,dZ_t,
$$

а шесть book-event intensities зависят от предсказуемого gap
$G_{t-}=M_{t-}-X_{t-}$ через one-sided linear ramps. Поэтому после book jump и до
следующего jump интенсивности не постоянны: их непрерывно меняет Brownian path $X$.
Обычный Gillespie step, который замораживает текущую total intensity до следующего
event, не является exact sampler этой модели. Глобального конечного dominating rate
также нет, поскольку intensities линейно растут с $|G|$.

Статья определяет continuous-time law, но не раскрывает generator code,
discretization, seeds или error control для Figure 4. P3 должен независимо
симулировать именно модель Definitions 2.1--2.2, не выдавая выбранную численную схему
за авторский или exact simulation algorithm. Кроме book events, continuous Brownian
movement может пересечь trading band между сохранёнными grid points; endpoint-only
проверка систематически пропускает такие fills и искажает passage time.

## Решение

### 1. Target law и название результата

**Exact model** означает только continuous-time law статьи:

- $X$ является непрерывным exogenous Brownian martingale;
- в tight state активны `slide_up`, `slide_down`, `open_up`, `open_down`, в open
  state --- `close_up`, `close_down`;
- slides меняют $M$ на $\pm\delta$, opens/closes --- на $\pm\delta/2$;
- conditional intensities вычисляются из predictable pre-event state;
- два book events не происходят одновременно почти наверное.

Реализация P3 называется **controlled numerical simulation of the exact model** или
`adaptive-frozen-intensity approximation`. Термины `exact simulation`, `exact
sampler` и `Gillespie simulation` для неё запрещены. Это различает exact
mathematical target и approximate numerical path. Статус Figure 4 остаётся
`independent partial reproduction`: выбранная semantics не восстанавливает
неизвестный авторский simulator.

Versioned algorithm identifier в `RunSpec v1`:
`adaptive_left_hazard_single_jump_v1`. Изменение event ordering, hazard rule или
crossing localization требует нового identifier и, после target run, protocol
amendment/new experiment ID.

### 2. State и вычисление intensities

На accepted step хранится один canonical state
`(t, x, mid_half_ticks, inventory, wealth_state)`, где `mid_half_ticks` --- integer
$k$ и $M=k\delta/2$. Gap всегда вычисляется как `g = m - x`, а не интегрируется как
независимая переменная. Parity получается как `k % 2`, spread однозначно выводится из
parity и не эволюционирует независимо; это исключает floating-point parity checks.

Для каждого шага все шесть $\lambda_j$ вычисляются в double precision из **левого**
state $(G_t,S_t)$. Inactive channels обязаны иметь нулевую intensity. Active
intensities должны быть конечными и неотрицательными; нарушение останавливает run,
а не исправляется clipping. Total intensity обозначается
$\Lambda_t=\sum_j\lambda_j(t)$.

### 3. Adaptive step и book-event approximation

Единственный resolution parameter --- dimensionless `epsilon`. Preregistered ladder:

```text
epsilon = 0.02   coarse diagnostic
epsilon = 0.01   primary
epsilon = 0.005  refinement
```

В paper-faithful balanced model reference rate равен
$\alpha_{ref}=2\alpha_s+\alpha_o=\alpha_c>0$. Для deliberately unbalanced control
`alpha_ref_per_second` задаётся явно в config и не выводится post hoc из результата.
Все regular observation times, calibration/activation boundaries и horizon endpoint
вычисляются из config до random draws. Пусть $t_{boundary}>t$ --- ближайшая такая
обязательная граница, либо $+\infty$, если её нет. Из left state выбирается

$$
h=\min\left\{T-t,t_{boundary}-t,\frac{\epsilon}{\alpha_{ref}},
\frac{-\log(1-\epsilon)}{\Lambda_t}\right\},
$$

где последний член считается $+\infty$ при $\Lambda_t=0$. Следовательно, для
замороженной left intensity

$$
p_{event}=1-e^{-\Lambda_t h}\le\epsilon.
$$

Шаг выбирается до любых новых random draws. Нельзя отвергать и пересэмплировать
Brownian increment из-за увиденного endpoint gap/intensity: такая anticipative
адаптация меняет law. Endpoint intensity и её изменение логируются как convergence
diagnostics, но не участвуют задним числом в выборе текущего $h$.

Следовательно, adaptive step никогда не пересекает requested observation timestamp:
он заканчивается точно на нём и использует обычный `brownian_increment` draw для
укороченного interval. Interpolation и дополнительные observation RNG draws запрещены.
Snapshot имеет phase rank после `end_state`, то есть видит Brownian endpoint, optional
book jump и post-jump strategy fill с этим timestamp. Schedule boundaries являются
частью `RunSpec`; их добавление меняет simulated approximation и требует нового config.

На шаге с probability $p_{event}$ выбирается **ровно один** book event, а с
probability $1-p_{event}$ --- ни одного. Условно на event channel выбирается с
probability $\lambda_j(t)/\Lambda_t$. Это exact first-event law только для rates,
замороженных в left state; схема отбрасывает второй и последующие jumps внутри шага.
При $\Lambda_t h=a$ соответствующая frozen-rate probability двух или более events
равна $1-e^{-a}(1+a)$ и убывает как $O(\epsilon^2)$ на шаг, но accumulated bias не
считается доказанно равным этому порядку. Error control даёт только refinement gate.

### 4. Fixed within-step ordering и timestamps

Каждый interval $(t,t+h]$ имеет неизменный phase ordering:

1. из pre-step state вычислить intensities и $h$, затем независимо выбрать optional
   book channel по frozen-left law;
2. провести Brownian evolution $X_t\to X_{t+h}$ при фиксированном $M_t$;
3. обработать в chronological order все обнаруженные continuous band crossings в
   $(t,t+h)$; они используют текущий, ещё не изменённый book state;
4. присвоить Brownian endpoint timestamp $t+h$;
5. если book event выбран, применить его jump к $M$ и parity в timestamp $t+h$;
6. из post-jump $(G_{t+h},S_{t+h})$ вычислить target inventory и немедленно исполнить
   необходимый order at the **post-jump touch**;
7. записать post-fill end state.

Если $t+h$ является mandatory observation boundary, после шага записывается immutable
`observation_snapshot`; он не меняет state и не расходует RNG.

Book event численно помещается в правый конец шага. Это approximation, а не
утверждение о true event time. Для полного порядка записи с одинаковым physical
timestamp получают monotone `sequence_id` и phase rank
`brownian_crossing_fill < book_jump < post_jump_strategy_fill < end_state < observation_snapshot`.
Brownian
crossing, локализованный только до правой границы interval, считается произошедшим в
`t+h^-`, то есть до book jump. Между book events $M$ и spread постоянны; book jump не
создаёт разрыва $X$.

При continuous crossing fill gap равен соответствующей границе $\pm\theta$ с
заданной localization tolerance, а half-spread берётся из текущего pre-jump book.
При book-jump crossing сначала полностью применяется price/parity transition, затем
fill использует post-jump gap и post-jump half-spread. Именно эта semantics сохраняет
paper convention $\phi_F$ at fill и измеримый jump overshoot. Никакой trade не меняет
book intensities: trader остаётся one-lot price taker модели статьи.

### 5. Brownian band crossings

Endpoint sign check недостаточен. После first entry у policy активна только одна
opposite boundary. Для Brownian segment с fixed $M$, у которого оба gap endpoints
$g_0,g_1<b$ для upper boundary $b$, exact conditional hit probability равна

$$
P\!\left(\sup G_s\ge b\mid g_0,g_1\right)
=\exp\!\left[-\frac{2(b-g_0)(b-g_1)}{\sigma_X^2h}\right],
$$

с симметричной формулой для lower boundary; endpoint за boundary означает certain
hit. Формулы применяются только к continuous part при фиксированном $M$.

#### 5.1. One active boundary после entry

Сначала одним draw решается exact hit/no-hit event по формуле выше. Conditioned hit
локализуется left-first recursion. Для parent segment sampled midpoint $Y$ предлагается
из unconditional Brownian-bridge Gaussian. По двум child endpoints вычисляются
$p_L,p_R$ и $p_U=1-(1-p_L)(1-p_R)$. Proposal $Y$ принимается с probability $p_U$;
это rejection sample midpoint conditional on parent hit, поскольку
$E[p_U]=p_{parent}$. После acceptance left child содержит first hit с conditional
probability $p_L/p_U$; иначе left child conditioned no-hit, а first hit находится в
right child. Выбранный child рекурсивно обрабатывается тем же правилом. Draw order и
left-first choice фиксированы; rejection attempts не имеют result-dependent cap.

Recursion останавливается при leaf width не больше `epsilon * h`. Если leaf endpoints
straddle boundary, hit time задаётся linear interpolation и лежит внутри leaf (либо на
совпадающем endpoint). Для bridge-only hit, когда оба endpoints остаются с одной
стороны, hit node ставится в строгий temporal midpoint leaf. В обоих случаях
$X_{hit}=M\mp\theta$ и $G=M-X$; independently changing only `gap` запрещено. При hit
ровно в сохранённом endpoint используется тот же $(t,X)$ node, поэтому двух значений
$X$ в один timestamp нет. Не семплированный positive-duration remainder является
обычным bridge от hit node к сохранённому endpoint. Это controlled localization
approximation; timestamp error не больше leaf width.

После fill активной становится opposite boundary. Необработанный remainder leaf и
сохранённые chronological sibling segments проверяются против новой boundary. Поэтому
несколько flips внутри исходного step разрешаются последовательно, но каждый момент
имеет только одну active boundary; independent competing one-sided tests запрещены.

#### 5.2. Единственный flat first entry

Flat state имеет две competing boundaries, для которых независимые one-sided Bernoulli
draws не задают joint first-exit law. Поэтому до первого entry применяется отдельная
явная approximation: для каждого step полностью семплируется dyadic Brownian-bridge
tree depth

$$
d=\left\lceil\log_2(1/\epsilon)\right\rceil,
$$

midpoints генерируются level-order, left-to-right. Полученный continuous piecewise-
linear path обходится chronologically; первая из $-\theta,+\theta$ пересечённых границ
определяет entry, а crossing time получается linear interpolation. Между dyadic nodes
дополнительный hidden excursion не добавляется. Это quantified fine-tree
approximation с maximum leaf width $h/2^d\le\epsilon h$, а не два независимых hit
tests. После entry остаток уже sampled polyline этого step не пересэмплируется и
обходится chronologically с новой opposite boundary. В symmetric flip policy flat entry бывает один раз, выполняется до policy
burn-in и не входит в measured renewal intervals; тем не менее его bias проходит
epsilon refinement и symmetry tests.

Bridge routine обязана пройти tests против analytic one-sided hit probability,
conditional-recursion distribution, flat-entry symmetry, chronological multiple flips,
continuity и deterministic replay. В artifacts сохраняются counts endpoint-detected,
bridge-only, flat-tree entries, rejection attempts и multiple-crossing refinements.
Localization и flat-tree truncation проходят общий resolution refinement; ни одна из
них не объявляется exact first-passage sampler.

### 6. Random streams и replay

RNG engine для P3 --- NumPy `PCG64DXSM`. Каждый replication имеет объявленный master
seed и независимые именованные streams как минимум:

- `brownian_increment`;
- `book_occurrence`;
- `book_channel`;
- `brownian_bridge`.

Substreams выводятся через `SeedSequence` и фиксированное versioned mapping имён, а не
через Python `hash()`, process ID или worker number. Добавление bridge draws не должно
сдвигать book-event streams; parallel scheduling не меняет seed-to-replication
mapping. Manifest сохраняет engine, NumPy version, master seed, ordered stream map и
algorithm version. Bitwise replay требуется в том же зафиксированном environment;
между версиями NumPy гарантируется только statistical contract, если upstream не
обещает иного.

В multi-policy sweep trader не влияет на endpoint/book skeleton, поэтому streams
`brownian_increment`, `book_occurrence` и `book_channel` можно разделять между
policies. Однако policy-dependent bridge refinement с одним последовательным stream
не задаёт общий continuous Brownian path. До появления policy-independent addressable
Brownian tree каждый policy получает domain-separated stream
`brownian_bridge/<policy_id>` из stable integer ID, а не из позиции runtime loop. Такой
run называется shared **market skeleton**, а не common-random-number или shared
continuous-path comparison. Inference resamples полный seed-cluster vector, сохраняя
общую skeleton dependence; scheduling или перестановка execution loop не меняют draws.

Одинаковые master seed IDs можно использовать на разных resolutions для удобства
аудита, но comparison не считается paired/common-random-number estimate без
отдельной доказанной coupling implementation. Adaptive draw counts сами по себе не
создают pathwise coupling.

### 7. Refinement и acceptance

Confirmatory result сначала считается при `epsilon = 0.01`, затем без изменения
primitives, seeds, burn-in, physical horizon, estimators или thresholds повторяется
при `epsilon = 0.005`. `epsilon = 0.02` служит coarse diagnostic. P3 gate проходит,
только если сдвиг каждой preregistered primary estimate между `0.01` и `0.005` не
превышает tolerance или Monte Carlo uncertainty, заранее заданные в synthetic
protocol. Несогласие означает unresolved numerical bias: horizon/seed count можно
увеличить для precision, но scientific parameters и gate нельзя подгонять; при
необходимости добавляется ещё более fine resolution через dated amendment.

Для каждого run сохраняются distribution шага $h$, maximum
$1-e^{-\Lambda_{left}h}$, quantiles $\Lambda_{right}h$, step/event counts, fraction
steps with event, bridge-only crossing rate, multiple-crossing refinements, runtime и
invariant violations. Cap относится только к frozen left rate; Brownian-driven rise
of right/interior intensity является известным residual error и оценивается
refinement, а не скрывается.

### 8. Pathwise invariants: `jump_simulator_invariants_v1`

Versioned suite `jump_simulator_invariants_v1` обозначает весь список ниже. Simulator
останавливает run и сохраняет failing pre/post record при любом нарушении:

- finite nonnegative active intensities, zero inactive intensities;
- strictly increasing numerical step time и полный порядок равных timestamps через
  phase/sequence;
- не более одного book jump на numerical step и ровно один выбранный channel;
- slides допустимы только в tight state, opens переводят tight в open, closes --- open
  в tight;
- $\Delta M\in\{\pm\delta,\pm\delta/2\}$ согласно channel и ни одного иного jump;
- parity $M/(\delta/2)$ согласована со spread $S\in\{\delta,2\delta\}$ после каждого
  transition;
- $X$ непрерывен на book event, $M$ неизменен на Brownian segment, $G=M-X$;
- continuous fill находится на $\pm\theta$ в localization tolerance; jump fill
  следует только после полного post-jump state update;
- target inventory лежит в $\{-1,+1\}$ после first entry, первый order равен one lot,
  каждый последующий flip --- two lots;
- overshoot band crossing by a legal book jump меньше $\delta$;
- при включённом accounting выполняется paper identity $W-W^X=qG$ в объявленной
  floating-point tolerance;
- один и тот же config/seed/environment даёт deterministic replay.

Balanced drift, stationary moments, occupancy и ACF являются statistical theorem
checks, а не pathwise invariants; clipping или state repair ради их прохождения
запрещены.

Book/state/numerical invariants применяются ко всем simulator runs. Crossing и target-
inventory invariants применяются только при `strategy_monitoring_enabled=true`;
wealth identity --- только при accounting enabled. Disabled components записываются
`not_applicable` с reason, а не как искусственный zero pass. Изменение состава или
applicability suite требует нового identifier.

## Отклонённые альтернативы

1. **Gillespie with current rate до следующего event.** Отклонено: Brownian $X$
   меняет intensities между events, поэтому exponential waiting time с frozen current
   rate не имеет target law.
2. **Fixed time grid без event-probability cap.** Отклонено: доля пропущенных multiple
   events зависит от параметров и остаётся неаудируемой.
3. **Независимый Bernoulli draw для каждого channel.** Отклонено: допускает
   simultaneous и mutually illegal parity transitions внутри одного step.
4. **Только endpoint threshold checks.** Отклонено: Brownian bridge может пересечь и
   вернуться за один step, что смещает first-passage estimates.
5. **Reject/resample большого Brownian endpoint.** Отклонено: решение, принятое после
   просмотра increment, создаёт selection bias, если не реализован корректный
   Brownian-tree stopping scheme.
6. **Global thinning bound.** Пока отклонено: для linear ramps на unbounded $G$ нет
   конечного global bound. Local exact thinning можно принять отдельным ADR после
   доказательства dominating construction и проверки против этой baseline scheme.
7. **Называть Figure 4 exact reproduction.** Отклонено: author primitives, seeds и
   simulator semantics не опубликованы.

## Последствия и риски

- P3 получает однозначный и реализуемый baseline, event log ordering и refinement
  gate; invariant и bridge tests можно написать до full Monte Carlo.
- Adaptive steps могут быть дорогими при больших $|G|$ или высоких baselines. Runtime
  является результатом, но не основанием ослаблять resolution после просмотра.
- Для схемы здесь не заявлен formal weak/strong convergence order. Frozen intensity,
  right-end event placement, omitted multiple jumps и approximate crossing time дают
  bias, который считается контролируемым только после empirical refinement gate.
- Left-rate cap не ограничивает pathwise interior intensity Brownian bridge. Endpoint
  diagnostics и halving `epsilon` обязательны; отсутствие observed discrepancy не
  является математическим доказательством exactness.
- Multiple Brownian flips внутри короткого step редки в целевом regime, но не
  объявляются невозможными; bridge routine обязана разрешать их chronological order.
- Любая будущая более точная event-time construction сравнивается с этим baseline на
  одних preregistered estimands. Замена primary simulator после просмотра Figure 4
  требует dated protocol amendment и нового experiment ID.
