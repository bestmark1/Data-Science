Коротко: в концепции правильные акценты — point-in-time, leakage, evidence-first, разделение offline/online, детерминированные вычисления, экспериментальная проверка. Но архитектурно это не платформа, а попытка заменить отсутствующую зрелость данных, метрик и процессов универсальным agent-framework с 60 модулями. В таком виде она будет дорогой, медленной, рискованной и, скорее всего, не будет использоваться.

---

## 1. Главные архитектурные и методологические ошибки

### 1.1. «Одно переиспользуемое ядро для всех DS-проектов» — ложная абстракция

Концепция предполагает, что проекты отличаются только конфигурацией: Domain Pack, Risk Profile, Metric System, Tools, Memory. Это неверно.

Проекты отличаются не параметрами, а:

- бизнес-решениями;
- ценой ошибки;
- доступностью данных;
- юридическими ограничениями;
- операционными процессами;
- моментами принятия решений;
- feedback loops;
- качеством и семантикой меток;
- возможностью рандомизации;
- допустимостью автономности.

Это нельзя выразить только конфигом. Конфигурация работает для узкого класса задач: табличные модели, типовые A/B-эксперименты, повторяющаяся продуктовая аналитика. Она не работает для регулируемых, медицинских, кредитных, страховых, производственных и многих операционных задач.

Правильный вывод: reusable может быть не «ядро интеллекта», а инфраструктурные примитивы и контрольные механизмы: доступ к данным, PIT-проверки, eval harness, журналирование, экспериментальный адаптер, аудит.

---

### 1.2. Domain Packs — это не программный компонент, а консалтинговая библиотека

20 Domain Packs — это не масштабируемая архитектура, а 20 продуктовых/доменных методологий. Каждый pack требует:

- SME;
- актуализации;
- юридических ограничений;
- метрик;
- типовых ошибок;
- данных;
- playbooks;
- регуляторных требований.

Если делать их честно, это огромная постоянная команда доменных экспертов. Если делать нечестно, получатся поверхностные шаблоны, которые будут вредить: навязывать неправильные KPI, target, leakage-риски, эксперименты.

Особенно опасно использовать Domain Pack как priors для LLM в регулируемых доменах. Ошибочный prior в healthcare, credit, insurance или employment может стать систематическим bias.

Domain Pack должен быть документацией, справочником и набором eval-кейсов, но не «модулем платформы», который автоматически влияет на поведение агента.

---

### 1.3. Lifecycle из 28 стадий превращается в бюрократический waterfall

Реальная аналитика и DS — итеративны. Вопрос меняется после первого взгляда на данные. Target меняется после проверки label quality. KPI меняется после обсуждения с бизнесом. Данные оказываются недоступны. Гипотеза отвергается до моделирования.

Жесткий lifecycle:

- создает иллюзию контроля;
- заставляет пользователей заполнять артефакты ради гейтов;
- тормозит time-to-value;
- плохо обрабатывает возвраты и переосмысление;
- провоцирует shadow usage: аналитик пойдет в notebook/SQL, а платформу будет использовать только для финального отчета.

Ветвление по типу задачи не спасает. Оно превращает lifecycle в дерево исключений. Нужен не lifecycle на 28 стадий, а короткий набор обязательных контрольных точек: problem definition, data readiness, baseline, evaluation, experiment/approval.

---

### 1.4. Data Readiness Gate и PIT как blocking layer могут остановить большинство проектов

Point-in-time correctness — правильная идея. Но как универсальный блокирующий слой она нереалистична.

Причины:

- исторические данные часто не имеют надежных `available_at`;
- витрины перезаписываются;
- CDC есть не везде;
- снапшоты не хранятся;
- бизнес-процессы менялись;
- label windows пересекаются с будущими событиями;
- агрегаты сложно восстановить на момент предсказания;
- ретротесты дороги.

Если система будет честно блокировать проекты без PIT-гарантий, она заблокирует много реальных задач. Если будет разрешать exceptions, ответственность размоется. Если будет выдавать warnings, их начнут игнорировать.

PIT должен быть не универсальным гейтом, а риск-ориентированным аудитом: где-то нужен hard block, где-то manual sign-off, где-то запрет ML и переход к описательной аналитике.

---

### 1.5. Leakage Validator не может быть полным, но концепция делает его критическим контрольным механизмом

Автоматически детектировать leakage невозможно полностью. Можно ловить часть типовых ошибок:

- future features;
- target-derived columns;
- неправильные временные окна;
- train/test contamination;
- entity leakage;
- preprocessing на полном датасете.

Но семантический leakage часто виден только domain expert:

- признак является следствием процесса, начинающегося после decision moment;
- поле обновляется задним числом;
- агрегат использует будущие статусы;
- operational system логирует результат уже после ручного вмешательства;
- метка сама является частью treatment.

Поэтому leakage validator не должен создавать ложное чувство безопасности. Он должен быть чеклистом и генератором вопросов к человеку, а не главным арбитром.

---

### 1.6. Gold Set + continuous learning — противоречие

Gold Set объявляется замороженным, независимым, защищенным от contamination. Но continuous learning постоянно производит новые метки, feedback, corrections, learned knowledge, experiment results.

Возникают проблемы:

- Gold Set устаревает;
- новые данные могут быть распределением уже не из Gold Set;
- learned knowledge может неявно использовать информацию из Gold Set;
- human corrections могут просачиваться в обучение;
- active learning меняет распределение размеченных данных;
- frozen Gold Set перестает отражать prod.

Нельзя одновременно иметь полностью замороженный Gold Set и непрерывное обучение. Нужны версии Gold Set, периодическая ротация, отдельные holdout cohorts и strict lineage. Это тяжело и дорого.

---

### 1.7. Multi-agent и reviewer model не дают независимости

Идея «analyst — модель A, reviewer — модель B» звучит надежно, но слабые места:

- модели могут иметь общие систематические ошибки;
- reviewer без детерминированной проверки фактов часто становится генератором правдоподобных возражений;
- multi-agent увеличивает стоимость, латентность и сложность debugging;
- orchestration errors часто важнее model errors;
- независимость моделей иллюзорна, если они обучены на похожих данных и используют один semantic layer.

Для high-risk задач нужен не LLM-reviewer, а:

- детерминированные тесты;
- human review;
- separate data validation;
- statistical checks;
- experiment;
- аудит.

LLM-reviewer может быть полезен как draft critic, но не как контрольный механизм.

---

### 1.8. Continuous Learning в аналитике опасен без causal identification

Концепция предполагает обучение из:

- traces;
- human feedback;
- corrections;
- failures;
- experiments;
- бизнес-результатов;
- новых меток;
- drift;
- находок reviewer’а.

Проблема: большая часть этих сигналов — observational, noisy, delayed, confounded, selection-biased.

Примеры опасных «выученных» знаний:

- «сегмент X хуже конвертируется» — но сегмент получился из-за политики скоринга;
- «эта фича важна» — но она leakage;
- «гипотеза A обычно работает» — но она проверялась только в высокий сезон;
- «метрика Y растет» — но изменился трекинг;
- «reviewer часто отклоняет выводы про churn» — и система научится избегать правильных, но неудобных выводов.

Continuous learning без строгой валидации быстро превращается в накопление суеверий. В регулируемых доменах он дополнительно конфликтует с change management, model risk management и аудитом.

Правильная версия: не continuous learning, а periodic supervised updates с versioned evals, human approval и rollback.

---

### 1.9. «Полный trace до бизнес-результата» недооценивает attribution problem

Проследить цепочку:

> вопрос → SQL → вывод → модель → решение → бизнес-KPI

в реальности очень сложно.

Причины:

- бизнес-результат delayed;
- на результат влияют внешние факторы;
- решение может быть изменено человеком;
- один вывод влияет на несколько действий;
- метрики могут быть переопределены;
- модель может быть только частью процесса;
- agent может использоваться как справочник, а не как decision-maker.

Полная причинно-следственная трассировка невозможна без сильных допущений. Нужно ограничиться audit trail до артефактов и решений, а business impact оценивать через эксперименты или quasi-experimental design, а не через trace.

---

### 1.10. Концепция пытается владеть всем сразу

Платформа одновременно хочет быть:

- project management;
- semantic layer;
- data quality;
- feature store;
- labeling system;
- MLOps;
- experiment platform;
- knowledge base;
- agent runtime;
- eval framework;
- learning engine;
- observability system;
- governance layer;
- compliance layer.

Это слишком много. Такой scope означает годы интеграций, дублирование существующих систем и вечный «platform build mode». Нужно не владеть всем, а встраиваться в существующий stack и добавлять недостающий контроль.

---

## 2. Ложные предпосылки, особенно H1 и H7

### H1: существует переиспользуемое ядро для существенно разных проектов

Скрытые предпосылки:

1. DS-задачи можно типизировать достаточно полно.
2. Доменные знания можно вынести в конфигурацию.
3. Semantic layer можно универсально описать для разных доменов.
4. Risk profile можно задать декларативно.
5. Пользователь согласится жить внутри platform lifecycle.
6. Ошибки конфигурации будут безопаснее, чем ручная работа.
7. Экономия на повторном использовании превысит стоимость поддержки ядра.

Почему это ложно:

- разные домены имеют разные causal structures;
- разные данные имеют разную временную семантику;
- разные бизнес-метрики имеют разную политику;
- разные регуляторные ограничения нельзя свести к одному risk engine;
- domain knowledge — это контент, а не код;
- универсальные priors могут ухудшать качество;
- стоимость поддержки 20 domain packs и 28-stage lifecycle выше, чем экономия.

Reuse есть, но он другой:

- reusable: SQL sandbox, PIT checks, leakage checklist, eval harness, experiment adapter, audit log;
- partially reusable: metric templates, playbooks, risk checklists;
- not reusable: business framing, target definition, causal assumptions, label semantics, decision policy, regulatory approval.

Вывод: H1 можно проверять только внутри одного домена и 2–3 похожих use cases, а не через «ядро для 20 доменов».

---

### H7: continuous learning может быть контролируемым, версионированным и обратимым

Скрытые предпосылки:

1. Есть достаточно качественных feedback-сигналов.
2. Feedback можно отделить от confounders.
3. Learning candidate можно валидировать дешевле, чем ручная работа.
4. Есть владелец знаний, который будет одобрять изменения.
5. Изменения можно безопасно откатывать.
6. Регуляторы и аудит примут versioned learned knowledge.
7. Knowledge updates не будут drift’ить в сторону удобства агента, а не бизнеса.

Почему это ложно:

- feedback часто редкий, delayed, biased;
- human corrections не являются ground truth;
- reviewer findings не являются валидированным знанием;
- experiment results часто не переносятся на другие сегменты и периоды;
- learned knowledge требует provenance, confidence, scope, owner, expiry — это дорого;
- в регулируемых доменах любое изменение behavior может требовать revalidation;
- если каждое изменение требует human approval, continuous learning становится bottleneck;
- если approval ослабить, система накапливает ошибки.

Правильная замена H7: «production failures and validated experiment results periodically become eval cases, documentation updates and model retraining candidates, with human approval».

Это не continuous learning, а controlled knowledge lifecycle.

---

## 3. Где переусложнение

### Можно удалить или сильно отложить

1. **20 Domain Packs.**  
   Нужен один домен и максимум один-два use case для MVP. Domain Pack должен быть документом и eval set, не runtime-компонентом.

2. **Multi-agent orchestration как отдельная архитектура.**  
   Для MVP достаточно single agent + deterministic validator + human approval. Multi-agent нужен только после доказанного выигрыша.

3. **Independent reviewer model.**  
   Пока нет доказательств, что вторая модель дает независимую проверку, это лишний расход и ложная уверенность.

4. **Learning Engine с девятью петлями обучения.**  
   Это overengineering. Сначала нужны: failure capture, regression evals, versioned docs, model retrigger.

5. **Labeling platform.**  
   Не нужно строить свою labeling system. Лучше интеграция с внешним labeling tool или ручные метки через существующие процессы.

6. **Project Memory / Learned Knowledge как отдельные сложные подсистемы.**  
   Пока достаточно versioned documents, decision log, eval cases и incident notes.

7. **Experimentation engine.**  
   Не нужно строить свой экспериментальный движок. Нужен адаптер к Statsig/GrowthBook/Eppo/внутренней системе.

8. **Retrotests как универсальный слой.**  
   Это дорого и нужно только для критичных PIT-задач. Не должно быть обязательным компонентом.

9. **Structural Break Detection как отдельный advisor.**  
   Может быть частью EDA и data monitoring, не отдельным модулем.

10. **Data Sufficiency Advisor как отдельный компонент.**  
    Learning curves и power analysis могут быть частью evaluation/experiment tools.

11. **Seasonality Advisor как отдельный advisor.**  
    Это набор проверок и визуализаций, не отдельный агентный модуль.

12. **Multimodal extension point.**  
    Не нужен в MVP вообще.

13. **Execution Router на основе complexity, uncertainty, risk, cost of error, latency, historical failure rate.**  
    Для MVP достаточно простых правил: high-risk → human review; low-risk → single agent; data mutation/PIT risk → block or manual approval.

14. **Полный observability trace до business KPI.**  
    Нужен audit log артефактов, запросов, версий, approvals. Business impact attribution — отдельно через эксперименты.

---

### Что схлопнуть в один компонент

1. **Project Config, Business Context, Risk Profile, Domain Pack**  
   → один Project Manifest с обязательными полями и human owner.

2. **Metric System, Semantic Layer**  
   → единый Metric & Semantic Catalog. Не отдельная платформа, а управляемый справочник с версиями, owner, valid_from/valid_to.

3. **Data Sources, Data Contracts**  
   → connectors + existing data catalog + ownership. Не нужно строить data contract system с нуля.

4. **Knowledge Base, Project Memory, Learned Knowledge, Experiment Knowledge**  
   → versioned knowledge repository + eval cases + decision log. Разделение на пять типов знания преждевременно.

5. **Tools, Models, Playbooks**  
   → Tool Registry + Model Registry + Runbook/Playbook documents.

6. **Все advisor’ы**: Domain Advisor, Business Problem Advisor, Metric Advisor, Semantic Advisor, Target Advisor, Data Requirements Advisor, Data Availability Advisor, Split Advisor, Seasonality Advisor, Structural Break Advisor, Data Sufficiency Advisor, Model Metric Advisor, Experiment Advisor  
   → один Project Setup Workflow с детерминированными формами, подсказками и LLM-draft explanations. Не нужно 13 отдельных advisor-модулей.

7. **Data Readiness Layer со всеми подмодулями**  
   → один Data Readiness Audit: profile, temporal checks, leakage checklist, PIT questionnaire, manual sign-off.

8. **Evaluation + Agent Evals + Business Impact Evals**  
   → один Eval Harness с разными типами тестов: golden Q&A, SQL correctness, leakage regression, metric calculation, experiment interpretation.

9. **Orchestration / Execution / Observability**  
   → application layer + logs + traces. Для MVP не нужно выделять это в сложную платформу.

---

## 4. Что должно быть детерминированным кодом, а что реально требует LLM

### Обязательно детерминированным кодом

1. **Выполнение SQL и Python в sandbox.**  
   LLM не должна выполнять произвольные вычисления вне контролируемого tool layer.

2. **Расчет метрик.**  
   KPI, model metrics, business metrics, guardrails — только по утвержденным формулам.

3. **Проверки качества данных.**  
   Nulls, duplicates, schema violations, row counts, date coverage, entity consistency.

4. **PIT checks.**  
   Проверка временных полей, snapshot availability, mutable tables, as-of joins, valid_from/valid_to.

5. **Leakage rules.**  
   Формализуемые проверки: future data, target-derived features, train/test overlap, entity leakage, preprocessing leakage.

6. **Split generation.**  
   Random, stratified, group, time-based, rolling, out-of-time — детерминированно и логируемо.

7. **Power analysis, sample size, MDE.**  
   Только формулы и симуляции, не LLM.

8. **Statistical tests.**  
   T-test, bootstrap, permutation tests, CUPED, difference-in-differences, если поддерживается, — детерминированные библиотеки.

9. **Baseline model training.**  
   Чтобы избежать произвольности и обеспечить воспроизводимость.

10. **Offline evaluation.**  
    Метрики, confusion matrix, calibration, lift, error analysis — детерминированно.

11. **Experiment analysis.**  
    Assignment, exposure, metric calculation, guardrail checks, report generation — детерминированно.

12. **Access control, allowlists, table permissions.**  
    Никакого LLM-решения «можно ли доступ».

13. **Versioning.**  
    Metrics, prompts, datasets, models, evals, configs — versioned artifacts.

14. **Regression evals.**  
    Запуск тестов после изменений должен быть детерминированным.

15. **Audit logs.**  
    Кто, что, когда, какой версии, какой инструмент вызвал, какие данные использовал.

---

### LLM реально нужен

1. **Черновик problem framing.**  
   Помочь сформулировать бизнес-вопрос, гипотезы, возможные KPI. Только как draft, не как источник истины.

2. **Перевод вопроса в план анализа.**  
   Какие срезы посмотреть, какие сравнения сделать, какие временные окна проверить.

3. **Генерация SQL/Python для review.**  
   LLM предлагает запрос, детерминированный слой проверяет синтаксис, таблицы, поля, фильтры, временные окна.

4. **Интерпретация результатов.**  
   Объяснить наблюдения, предложить альтернативные гипотезы, указать ограничения.

5. **Draft critique.**  
   Найти unsupported claims, causal overclaiming, missing caveats. Но не финальный review.

6. **Суммаризация evidence.**  
   Собрать отчет: вопрос, данные, проверки, результаты, ограничения, рекомендации.

7. **Извлечение кандидатов в знания.**  
   Из инцидентов, feedback, experiment results — только как candidate, не как automatic update.

---

### Где LLM вставлен без нужды

1. **Выбор KPI.**  
   KPI должен выбирать бизнес-владелец. LLM может предлагать варианты, но не определять.

2. **Утверждение target.**  
   Target — это продуктовое/операционное/юридическое решение. LLM не должен быть target advisor в финальном смысле.

3. **Вердикт data readiness.**  
   Может быть автоматический score, но решение READY/NOT READY для критичных задач должно быть human-gated.

4. **Финальное решение о leakage.**  
   LLM может указать риск, но не должен блокировать или разрешать самостоятельно.

5. **Расчет statistical significance.**  
   LLM не должен считать p-values, power, confidence intervals «в голове».

6. **Causal conclusion.**  
   LLM может формулировать гипотезы, но causal claim должен опираться на design и deterministic analysis.

7. **Learning updates.**  
   LLM не должен автоматически переписывать rules, semantic definitions, metric definitions, risk policies.

8. **Risk classification.**  
   Risk profile должен задаваться по правилам и ответственному лицу, а не выводиться LLM.

9. **Execution routing по «uncertainty» и «cost of error».**  
   Это сложно измерить надежно. Для MVP лучше простая policy-based маршрутизация.

10. **Reviewer как единственный защитный слой.**  
    LLM-reviewer не заменяет tests, human SME, experiment, audit.

---

## 5. Экономика и организационные предпосылки

### Чтобы платформа имела смысл, в компании должно быть верно следующее

1. **Много повторяющихся DS/analytics проектов.**  
   Если проектов 2–3 в год, платформа не окупится.

2. **Зрелый data warehouse.**  
   Есть таблицы, lineage, ownership, временные атрибуты, снапшоты или хотя бы понятная история изменений.

3. **Есть metric ownership.**  
   Кто-то владеет определениями метрик и может их утверждать. Иначе platform будет генерировать semantic conflicts.

4. **Есть experiment culture.**  
   Люди умеют и готовы запускать A/B, holdout, phased rollout. Если экспериментов нет, causal layer бесполезен.

5. **Есть compliance/audit requirement.**  
   Если регуляторного давления нет, heavy governance будет восприниматься как тормоз.

6. **Есть выделенная platform team.**  
   Не 2 человека между проектами, а команда с product owner, data engineer, ML engineer, eval engineer, domain SME.

7. **Есть бюджет на 12–18 месяцев до полезного scale.**  
   Такая платформа не дает ценности за один квартал.

8. **Пользователи готовы терять скорость ради контроля.**  
   В регулируемых доменах это возможно. В growth-командах — часто нет.

9. **Есть ответственность за business KPI.**  
   Если никто не отвечает за связь model metric → business outcome, платформа станет отчетной системой.

10. **Существующие инструменты не закрывают задачу.**  
    Если dbt + MLflow + GrowthBook + notebook уже работают, новая платформа должна давать явный выигрыш.

---

### Кто пользователь

Возможные пользователи:

- data scientist;
- analyst;
- analytics engineer;
- product manager;
- risk/compliance;
- business owner.

Но у них разные интересы:

- DS хочет гибкость;
- аналитик хочет скорость;
- PM хочет ответ;
- compliance хочет контроль;
- business owner хочет impact.

Платформа пытается угодить всем сразу и поэтому рискует не угодить никому.

---

### Почему пользователю не будет хотеться ею пользоваться

1. **Слишком высокий вход.**  
   Нужно заполнить Project Manifest, Metric System, Semantic Layer, Risk Profile, Data Readiness, Eval Set.

2. **Медленный time-to-first-answer.**  
   Пользователю нужен ответ сегодня, а платформа требует lifecycle.

3. **LLM-подсказки будут нести ответственность, но не имеют власти.**  
   Если совет плохой, пользователь теряет доверие. Если хороший — все равно нужно много ручного подтверждения.

4. **Гейты блокируют работу.**  
   Data Readiness Gate, leakage validator, risk review будут восприниматься как препятствия.

5. **Нет интеграции с привычным workflow.**  
   Если платформа не встроена в SQL IDE, notebook, BI, experiment tool, ей будут пользоваться «для галочки».

6. **Ответственность размыта.**  
   Если агент предложил KPI/target/вывод, кто отвечает за ошибку? Платформа? Пользователь? Reviewer?

7. **Semantic layer устаревает.**  
   Если метрики и определения не обновляются владельцами, агент начинает давать формально правильные, но бизнес-неверные ответы.

8. **Стоимость запроса выше ручной работы.**  
   Multi-agent, reviewer, evals, tracing, human approval могут сделать каждый анализ дороже, чем ручная работа опытного аналитика.

---

## 6. Конкурентная реальность

### Что уже существует

1. **dbt**  
   Трансформации, тесты, documentation, lineage, semantic layer в разных формах.

2. **Metrics layer / semantic layer**  
   dbt Semantic Layer, Cube, LookML, AtScale, локальные metric stores.

3. **Data quality**  
   Great Expectations, Soda, Monte Carlo, Anomalo, встроенные проверки warehouse.

4. **Feature store / PIT**  
   Feast, Tecton, Hopsworks, внутренние feature platforms.

5. **MLflow / W&B / Model Registry**  
   Эксперименты, артефакты, версии моделей.

6. **Experimentation platforms**  
   Statsig, GrowthBook, Eppo, Optimizely, LaunchDarkly, внутренние A/B systems.

7. **Observability / ML monitoring**  
   Arize, WhyLabs, Fiddler, Evidently, Grafana, Datadog.

8. **AI analysts**  
   Hex Magic, Databricks Assistant, Snowflake Cortex, ThoughtSpot Sage, Power BI Copilot, Julius, внутренние copilots.

9. **Eval platforms**  
   LangSmith, Braintrust, Humanloop, Arize Phoenix, локальные eval harnesses.

---

### Где новая ценность

Новая ценность не в том, чтобы заменить все эти системы. Она может быть в тонком governance-слое, который связывает:

- бизнес-вопрос;
- утвержденные метрики;
- data readiness;
- PIT/leakage checks;
- baseline;
- eval;
- experiment;
- decision;
- audit trail.

Это ценность не «агента», а controlled evidence chain.

Если платформа делает это легче, чем ручные процессы + существующие инструменты, у нее есть шанс. Если она пытается заменить dbt, feature store, labeling tool, experimentation platform и knowledge base одновременно — это переизобретение.

---

### Где переизобретение

1. **Experimentation engine.**  
   Statsig/GrowthBook/Eppo уже решают assignment, analysis, guardrails, feature flags.

2. **Labeling system.**  
   Scale, Label Studio, Argilla, внутренние labeling pipelines.

3. **Feature store.**  
   Feast/Tecton/внутренние системы лучше решают PIT feature serving.

4. **Data contracts.**  
   Лучше решаются data catalog, dbt contracts, ownership, schema registries.

5. **Observability.**  
   OpenTelemetry, MLflow, W&B, Datadog, specialized ML monitoring.

6. **Knowledge base/RAG.**  
   Обычно достаточно Confluence/Notion/wiki + search + access control. Специализированный RAG нужен только при большом объеме структурированных доменных знаний.

7. **Multi-agent runtime.**  
   Многие orchestration задачи решаются проще через workflow engine, tests, human review.

---

## 7. Радикально более простой MVP

Цель MVP — не построить платформу, а проверить, что controlled DS/аналитический процесс вообще дает выигрыш перед ручным workflow.

### MVP должен иметь 5 блоков

#### 1. Project Manifest

Человек заполняет и утверждает:

- бизнес-вопрос;
- владелец;
- риск-уровень;
- KPI;
- decision to support;
- target, если есть ML;
- data sources;
- known limitations;
- required approvals.

LLM может помогать черновиком, но не утверждать.

Проверяет: H2 частично, H1 в пределах одного домена.

---

#### 2. Metric & Semantic Catalog

Минимальный каталог:

- метрика;
- формула;
- owner;
- version;
- valid_from/valid_to;
- source table/fields;
- known caveats.

Никакого универсального semantic engine. Только утвержденные определения для одного-двух use cases.

Проверяет: H2, частично H6.

---

#### 3. Data Readiness Audit

Один сервис, который выполняет:

- profile таблиц;
- date range checks;
- null/duplicate checks;
- entity consistency;
- mutable table questionnaire;
- PIT checklist;
- leakage checklist;
- label availability check, если есть target;
- output: READY / NOT READY / CONDITIONAL с обязательными вопросами к владельцу данных.

Не нужно 10 отдельных advisors. Нужен один отчет и human sign-off.

Проверяет: H3.

---

#### 4. Single Analytics Agent

Один агент с read-only SQL/Python sandbox:

- предлагает план анализа;
- генерирует SQL/Python;
- выполняет только через tool layer;
- возвращает evidence table;
- пишет интерпретацию с ограничениями;
- не выносит final causal conclusions без experiment/human approval.

Никакого multi-agent, router, parallel analysis, independent reviewer model.

Дополнительно:

- deterministic SQL validator;
- schema/table allowlist;
- row count and period checks;
- metric calculation только по catalog formulas;
- все запросы логируются.

Проверяет: H2, H3, частично H5 через сравнение с ручным процессом.

---

#### 5. Experiment Adapter

Не свой experimentation layer, а адаптер к существующей системе:

- передает hypothesis, primary metric, guardrails;
- вызывает deterministic sample size/power calculator;
- получает результаты;
- сравнивает offline model metric и online experiment result;
- формирует report: offline improvement, online effect, guardrail impact, business interpretation.

Проверяет: H6.

---

### Что еще нужно в MVP

1. **Golden Eval Set.**  
   20–50 кейсов для одного домена: вопросы, ожидаемые SQL-паттерны, ожидаемые проверки, правильные ограничения, известные leakage traps.

2. **Failure Capture.**  
   Любая ошибка в pilot projects становится regression eval candidate. Без auto-learning.

3. **Audit Report.**  
   Версии метрик, данных, запросов, prompt version, model version, approvals, experiment reference.

---

### Какие гипотезы MVP проверяет

| Гипотеза | Проверяется MVP? | Как |
|---|---:|---|
| H1: reusable core для разных проектов | Частично и осторожно | Только внутри одного домена и 2–3 проектов |
| H2: business problem → KPI → DS problem → target | Да | Project Manifest + Metric Catalog + human review |
| H3: проблемы данных выявляются до моделирования | Да | Data Readiness Audit |
| H4: удешевление разметки без потери надежности | Нет | Откладывается, labeling не строим |
| H5: router выбирает сложность только когда окупается | Частично | Можно сравнить simple agent vs agent + human review, но без сложного router |
| H6: система отличает offline improvement от business effect | Да | Experiment Adapter + offline/online report |
| H7: controlled continuous learning | Нет честно | Только failure capture → regression eval candidate |
| H8: end-to-end learning cycle | Нет | Откладывается до подтверждения ценности базового цикла |

---

### Что MVP честно откладывает

1. 20 Domain Packs.
2. Multi-agent orchestration.
3. Learning Engine.
4. Labeling platform.
5. Full continuous learning.
6. Advanced causal inference.
7. Multi-model reviewer.
8. Project Memory как отдельная система.
9. Full business KPI attribution.
10. Autonomous updates semantic layer/metrics.
11. Universal PIT guarantee.
12. Regulatory certification.
13. Multimodal.
14. Custom experimentation engine.

Это не значит, что это никогда не нужно. Это значит, что сначала нужно доказать базовую ценность без этих компонентов.

---

## 8. Топ-5 рисков, которые убьют проект, и ранние сигналы

### Риск 1. Платформой не будут пользоваться

Суть: пользователи уйдут в notebook/SQL/BI, потому что платформа медленнее и тяжелее.

Ранние сигналы:

- пилотные команды просят exceptions, чтобы работать вне платформы;
- setup проекта занимает дни;
- пользователи заполняют поля формально;
- после первого анализа платформа не используется повторно;
- результаты переносятся в PowerPoint/Confluence вручную;
- нет повторных запросов от одного пользователя.

Митигация:

- сократить setup до 30–60 минут;
- встраиваться в SQL/notebook workflow;
- давать ценность до всех гейтов;
- измерять time-to-first-evidence.

---

### Риск 2. Data readiness/PIT/leakage gates будут либо блокировать всё, либо давать ложную уверенность

Суть: если гейты слишком строгие — платформа бесполезна; если слишком мягкие — платформа пропускает дефектные модели.

Ранние сигналы:

- большинство проектов получают NOT READY без понятного пути исправления;
- много manual overrides;
- warnings игнорируются;
- после «passed» обнаруживается leakage;
- domain experts не доверяют автоматическим проверкам;
- remediation занимает больше времени, чем modeling.

Митигация:

- риск-ориентированные гейты;
- обязательный human sign-off для high-risk;
- четкий список blocking vs advisory checks;
- пост-инцидентный анализ всех passed-but-wrong случаев.

---

### Риск 3. LLM будет давать убедительные, но ошибочные аналитические и causal выводы

Суть: ошибка LLM в SQL, метрике, сегменте, временном окне или интерпретации может выглядеть как экспертный вывод.

Ранние сигналы:

- пользователи находят ошибки в SQL после агента;
- агент путает metric definitions;
- появляются causal claims без эксперимента;
- validator ловит только синтаксис, а не смысл;
- reviewer model соглашается с плохим выводом;
- растет число stakeholder escalations.

Митигация:

- metric calculation только детерминированный;
- evidence table с обязательными ограничениями;
- запрет causal language без experiment/design;
- human approval для high-risk;
- golden evals на semantic errors, не только syntax.

---

### Риск 4. Continuous learning и governance станут несовместимыми

Суть: либо learning не проходит через approval и становится неконтролируемым, либо approval делает learning невозможным.

Ранние сигналы:

- очередь learning candidates без owner;
- learned knowledge не имеет confidence/scope/expiry;
- audit не может объяснить, почему агент дал рекомендацию;
- compliance требует заморозить изменения;
- изменения semantic layer требуют недель согласования;
- после инцидента невозможно откатить поведение агента.

Митигация:

- не называть это continuous learning;
- ввести periodic supervised releases;
- все knowledge changes через human approval;
- каждая версия behavior должна иметь eval report;
- learned knowledge хранить как candidate, не как runtime truth.

---

### Риск 5. Экономика не сойдется: стоимость платформы превысит стоимость ручного процесса

Суть: платформа может быть технически правильной, но слишком дорогой в эксплуатации.

Ранние сигналы:

- platform team растет быстрее, чем число активных проектов;
- каждый новый проект требует кастомных интеграций;
- поддержка domain packs занимает больше времени, чем разработка фич;
- latency/cost на один анализ выше ручной работы;
- бизнес не может назвать ни одного подтвержденного KPI improvement;
- платформу держат только из-за executive sponsorship, а не пользовательского спроса.

Митигация:

- считать cost per project;
- сравнивать с baseline manual process;
- требовать 3–5 pilot projects с измеримым сокращением времени или предотвращенными ошибками;
- не строить 20 доменов до успеха в одном;
- интегрировать existing tools вместо замены.

---

## Итоговая позиция

Концепция правильная как набор дисциплин, но плохая как архитектура продукта.

Нужно сохранить:

- point-in-time;
- leakage control;
- evidence before conclusion;
- deterministic calculations;
- separation of evals and business impact;
- experiment gating;
- auditability;
- human approval for high-risk decisions.

Нужно отвергнуть:

- универсальное ядро для 20 доменов;
- 28-stage lifecycle как основной процесс;
- multi-agent complexity по умолчанию;
- continuous learning как runtime mechanism;
- собственную labeling/experimentation/knowledge platform;
- belief, что LLM-reviewer обеспечивает safety;
- belief, что configuration решает domain complexity.

Правильный путь: не «Adaptive Analytics Agent Core», а **controlled evidence and decision gateway** поверх существующего data/ML/experiment stack. Минимум компонентов, максимум детерминированной проверки, один домен, один use case, human-owned semantics, experiment-based validation и eval-driven improvement без самообучения.