Ниже — аналитическое ревью. Главное: до сплита и baseline нужно переопределить таргет по календарным дням, пересчитать цензурирование и label lag, и явно решить, что делать с canceled/unavailable. Иначе positive_rate, embargo, split и выбор порога будут построены на другом estimand.

## Топ-5 по влиянию на результат

1. **[Критично] Таргет сравнивает datetime с date.**  
   `order_delivered_customer_date > order_estimated_delivery_date` считает опозданием доставку в обещанный день, если timestamp позже 00:00 обещанной даты. Для promesse «доставить до даты» доставка в ту же календарную дату — on_time. Это напрямую меняет число late, positive_rate, все leak-проверки и последующие метрики.

2. **[Критично] Правое цензурирование посчитано подозрительно узко.**  
   4 unobservable при horizonte обещанных дат в будущем выглядят подозрительно. Нужно проверить snapshot, границу дня и состав недоставленных. Если `order_estimated_delivery_date` бывает позже конца наблюдения, а среди последних заказов есть недоставленные, unobservable должно быть больше.

3. **[Критично] Label lag посчитан неправильно.**  
   У тебя это lag от покупки до фактической доставки для доставленных. Но для таргета «доставка позже обещанной даты» исход становится известен раньше:  
   - если доставлено вовремя — в `order_delivered_customer_date`;  
   - если не доставлено к обещанной дате — в `order_estimated_delivery_date`.  
   Поздняя фактическая доставка через 209 дней не нужна, чтобы знать, что заказ опоздал.

4. **[Критично] `late_never_delivered` смешивает опоздание и отмену/недоступность.**  
   Если в эту группу входят canceled, unavailable, invoiced, processing и т.п., ты предсказываешь не «доставка будет поздно», а «заказ не будет доставлен к обещанной дате по любой причине». Это другой бизнес-estimand. Модель может учиться предсказывать отмены и payment failure, а не логистические задержки.

5. **[Высоко] Часть «доступных» признаков может быть post-treatment.**  
   Особенно `shipping_limit_max` и признаки, связанные с отсутствием/наличием платежей и позиций. Их нужно либо доказать как known-at-decision, либо перевести в forbidden/ambiguous.

---

# 1. Ошибки в анализе

## 1.1 Определение late: нужна календарная дата, а не timestamp

**[Критично]**

Сейчас:

```python
late_delivered = order_delivered_customer_date > order_estimated_delivery_date
```

`order_estimated_delivery_date` после парсинга, скорее всего, становится датой в 00:00. `order_delivered_customer_date` — timestamp. Поэтому доставка `2018-01-10 14:00` при обещании `2018-01-10` считается late.

Правильная базовая формулировка:

```text
late = date(order_delivered_customer_date) > date(order_estimated_delivery_date)
```

или эквивалентно:

```text
on_time = date(order_delivered_customer_date) <= date(order_estimated_delivery_date)
```

После этого нужно пересчитать:

- counts on_time / late / late_never_delivered / unobservable;
- positive_rate;
- все эмпирические leak-разрывы;
- baseline;
- calibration.

Это меняет саму целевую переменную.

---

## 1.2 Правое цензурирование: 4 unobservable — подозрительно

**[Критично]**

У тебя:

```python
data_end = max(order_purchase_timestamp)
estimate_passed = order_estimated_delivery_date < data_end
```

Проблемы:

1. **Не подтверждено, что `max(order_purchase_timestamp)` — это конец наблюдения.**  
   Нужно проверить максимумы всех фактических событий:
   - purchase;
   - approved;
   - carrier;
   - delivered.  
   Estimated delivery — плановая будущая дата, не событие наблюдения.

2. **Граница дня.**  
   Если snapshot, например, `2018-10-17 17:30:00`, а обещанная дата `2018-10-17`, то день обещания ещё не закончился. Формула `estimated_date < data_end` может считать such orders уже late, хотя на момент snapshot исход ещё не наблюдался до конца дня.

   Безопаснее:

   ```text
   estimate_passed = estimated_date < snapshot_date
   ```

   если snapshot не является концом дня. Либо задать snapshot как конец дня.

3. **Подозрительно мало unobservable.**  
   Если `max(order_estimated_delivery_date)` позже snapshot и есть недоставленные заказы за последние дни/недели, unobservable должно быть больше.

Обязательные проверки:

```text
max(order_purchase_timestamp)
max(order_delivered_customer_date)
max(order_estimated_delivery_date)

число недоставленных с estimated_date >= snapshot_date
распределение estimated_date - purchase_date для последних 30 дней
состав этих заказов по order_status
```

Если после корректной дневной границы и правильного snapshot всё равно остаётся 4 unobservable — это возможно, но должно быть доказано, а не получено случайно из datetime-сравнения.

---

## 1.3 Label lag и horizon_days посчитаны не для таргета

**[Критично]**

Твой label lag:

```text
order_delivered_customer_date - order_purchase_timestamp
```

для доставленных заказов.

Но для предсказания «будет ли доставка позже обещанной даты» момент, когда исход становится известен:

```text
если доставлено вовремя:
    label_known_at = order_delivered_customer_date

если не доставлено к обещанной дате:
    label_known_at = order_estimated_delivery_date
```

То есть для late-заказов, которые фактически доставили через 100 или 209 дней, исход известен уже в обещанную дату, если к ней нет доставки.

Следствие:

- `horizon_days: 209` не является label horizon;
- это максимум фактической доставки, а не момент узнавания таргета;
- embargo нужно считать от `label_known_at - purchase_timestamp`, а не от фактической доставки;
- p95 label lag может быть другим.

Формула для пересчёта:

```text
label_known_at =
    delivered_date, если доставлено и date(delivered_date) <= estimated_date
    estimated_date, если заказ не был доставлен к estimated_date
```

unobservable:

```text
нет доставки и estimated_date >= snapshot_date
```

---

## 1.4 Трактовка never_delivered: estimand не зафиксирован

**[Критично]**

Ты отметил недоставленные с истёкшим сроком как late. Это допустимо, но только если целевой вопрос:

```text
заказ не будет доставлен к обещанной дате по любой причине
```

Если же вопрос:

```text
доставка будет позже обещанной даты
```

то canceled/unavailable/invoiced/processing — отдельная проблема. Это не обязательно «поздняя доставка». Это может быть «доставки не будет».

Нужно проверить состав `late_never_delivered` по `order_status`.

Если там много canceled/unavailable, то текущий positive class включает отмены. Тогда:

- positive_rate завышен относительно чистой логистической задержки;
- модель может ловить payment failure, cancellation, seller failure;
- action «предупредить о поздней доставке» может быть неприменим к отменённым заказам;
- нужен либо отдельный target, либо явное признание composite outcome.

Варианты:

1. **Composite failure-to-deliver-by-promise**:  
   late = delivered after promise OR not delivered by promise.  
   Тогда canceled/unavailable остаются в late, но в контракте нужно явно написать, что target — не чистая задержка, а недоставка к дате.

2. **Logistics delay only**:  
   исключить canceled/unavailable или считать их competing risk.  
   Но это сложно, потому что отмена — post-treatment событие, и исключение может создать selection bias.

3. **Две модели**:  
   - вероятность отмены;
   - вероятность поздней доставки среди неотменённых.  
   Но для этапа 0 это уже усложнение.

Сейчас в контракте эта развилка не закрыта.

---

## 1.5 Классификация доступности признаков

**[Высоко]**

В целом разумная, но есть несколько опасных мест.

### `shipping_limit_max`

Сейчас ambiguous. Для надёжности лучше считать forbidden, пока не доказано, что дата известна в момент оформления.

Причина: shipping limit может зависеть от:

- approval;
- seller handling time;
- платежного статуса;
- внутренней логики фулфилмента после покупки.

Если `shipping_limit_date` выставляется после `order_approved_at`, это post-treatment.

Проверка:

```text
shipping_limit_date - order_purchase_timestamp
order_approved_at - order_purchase_timestamp
корреляция между ними
```

Если limit часто привязан к approval, признак нельзя использовать как at_decision.

### Payment features

`payment_installments`, `payment_value`, `payment_types` — выбраны пользователем при оформлении, но наличие записи в payments может быть связано с успешностью оплаты.

Опасно:

- использовать только заказы с платежами через inner join;
- кодировать отсутствие платежа как признак;
- импутировать отсутствие платежа нулями без флага.

Отсутствие платежа может быть следствием отмены или payment failure после decision. Это post-treatment missingness.

### Items features

`items_count`, `items_total`, `freight_total`, `sellers_count` выглядят как характеристики корзины в момент покупки. Но заказы без позиций — 775 — нужно отдельно разобрать. Если отсутствие items связано с отменой/ошибкой, то импутация нулями может стать прокси для post-treatment статуса.

### `order_estimated_delivery_date`

Скорее всего at_decision, если это дата, обещанная покупателю при оформлении. Но нужно зафиксировать допущение:

```text
estimated_delivery_date назначается до или в момент purchase и не меняется после.
```

Если она назначается после выбора seller/shipping method уже после purchase, признак может быть post-treatment.

### `product_weight_g`, `product_volume_cm3`, category, product_id, seller_id

Сами по себе не leak: это атрибуты товаров/продавцов, известные в момент корзины. Но нужно определить агрегацию для multi-item заказов и не использовать будущие исторические признаки без as-of логики.

---

## 1.6 Эмпирическая проверка лика через квартили

**[Средне]**

Проверка рабочая как грубый скрининг, но недостаточная.

Ограничения:

1. Смотрит только низкие vs высокие квартили.  
   Может пропустить:
   - немонотонные зависимости;
   - категориальные признаки;
   - редкие категории;
   - leak через missingness.

2. `shipping_limit_max` как datetime, вероятно, не попал в numeric-проверку.  
   Его надо было бы переводить в дни от purchase, но сначала решить доступность.

3. Для признаков с большим числом одинаковых значений Q1 и Q3 могут совпадать, тогда сравнение теряет смысл.

4. Не проверяется сила связи как single-feature AUC / mutual information / lift.

5. Не проверяются пропуски как отдельный сигнал.  
   Например, отсутствие items или payments может быть сильнее любого числового признака.

Минимально нужное дополнение:

- single-feature AUC или rank-based score для числовых признаков;
- late rate по категориям для categorical;
- late rate для missing / not missing;
- проверка временных признаков: месяц, неделя, день недели;
- проверка lead time = estimated_date - purchase_date.

---

## 1.7 Верные выводы

Без существенных замечаний:

- временные метки как String;
- `customer_id` — order-customer key, человек — `customer_unique_id`;
- дубли в geolocation;
- нарушения порядка меток;
- 8 delivered без даты доставки;
- fanout при naive join и завышение суммы price;
- сироты по items/payments/reviews;
- review_score как post-outcome признак;
- `order_status` становится почти идеальным индикатором из-за твоей разметки never_delivered.

Но выводы по positive_rate, label lag, unobservable и embargo нужно пересчитать после исправлений выше.

---

# 2. Что пропущено в данных

Ниже — только то, что реально влияет на эту задачу.

## 2.1 Проверка estimated_date против purchase_date

**[Критично]**

Нужно проверить:

```text
estimated_date < purchase_date
estimated_date == purchase_date
estimated_date - purchase_date <= 0
```

Если есть отрицательный lead time, таргет ломается. Такие заказы нужно либо исключить, либо отдельно исследовать.

Также нужно посмотреть распределение:

```text
lead_time = estimated_date - purchase_date
```

по:

- месяцам;
- статусам;
- seller_state;
- customer_state;
- shipping method, если его можно восстановить косвенно.

Если lead time менялся во времени, out-of-time split может получить другой positive_rate.

---

## 2.2 Состав never_delivered по order_status

**[Критично]**

Для каждого статуса:

```text
delivered
shipped
canceled
unavailable
invoiced
processing
created
```

посчитать:

- число заказов;
- число с estimated_date passed;
- число unobservable;
- долю в positive class.

Особенно важно:

```text
сколько late_never_delivered приходится на canceled/unavailable;
сколько — на shipped/processing/invoiced.
```

Это меняет интерпретацию таргета.

---

## 2.3 Same-day deliveries

**[Критично]**

Нужно посчитать:

```text
доставлено в день estimated_date
доставлено на следующий день
доставлено до estimated_date
```

Это прямо покажет, насколько positive_rate завышен из-за datetime-сравнения.

---

## 2.4 Связка order_status и фактических дат

**[Высоко]**

Проверить:

- delivered, но нет `order_delivered_customer_date`;
- не delivered, но есть `order_delivered_customer_date`;
- canceled, но есть delivery date;
- shipped, но нет carrier date;
- delivered customer date раньше carrier date;
- delivered customer date раньше approved date.

Ты уже нашёл часть, но для таргета важно specifically:

```text
какие заказы с delivery date имеют статус, отличный от delivered;
как их трактовать.
```

Сейчас таргет использует наличие `order_delivered_customer_date`, игнорируя `order_status`. Это может быть правильно, но должно быть осознанным правилом.

---

## 2.5 Заказы без items и payments

**[Высоко]**

Для 775 заказов без items и 1 без payments нужно проверить:

- их `order_status`;
- delivered ли они;
- estimated passed;
- долю positive;
- являются ли они canceled/unavailable;
- что означает отсутствие items: ошибка, отмена, data issue.

Если они остаются в обучении, нужно определить:

```text
items_count = null? 0?
freight_total = null? 0?
payment_value = null? 0?
```

Импуляция нулём может создать leak-признак «заказ без позиций», который коррелирует с отменой.

---

## 2.6 Консистентность payments и items

**[Высоко]**

Проверить:

```text
payment_value_sum vs items_total + freight_total
```

Если суммы расходятся:

- что считать ценой заказа?
- есть ли скидки?
- есть ли частичные платежи?
- несколько платежей — это split payment или повторные попытки?

Для признаков нужно задать агрегацию:

```text
payment_value_sum
payment_installments_max / sum?
payment_types set / one-hot / dominant
```

Без этого модель может получить нестабильные признаки.

---

## 2.7 Geolocation duplicates и zip/city consistency

**[Средне/Высоко, если использовать geo]**

Если планируешь использовать `customer_zip_code_prefix`, city, state или lat/lon:

- проверить дубли zip prefix;
- выбрать стратегию dedup: first, mode, median lat/lon;
- проверить несоответствие zip -> state;
- решить, использовать ли `customer_state` из customers или geolocation.

Иначе join geolocation может создать fanout или случайные координаты.

---

## 2.8 Временной дрейф

**[Высоко]**

Нужно проверить:

- late rate по неделям/месяцам;
- late rate по кварталам;
- долю never_delivered по месяцам;
- средний lead time по месяцам;
- объём заказов по месяцам;
- сезонность, включая Black Friday и декабрь.

Если дрейф сильный, out-of-time split должен быть выбран осознанно, а baseline должен быть temporal, а не global.

---

## 2.9 Сильные легитимные признаки, которых пока нет

**[Высоко для будущей модели]**

Для этой задачи сильными законными at_decision признаками могут быть:

- seller_id;
- product_id;
- product category;
- seller_state / seller_zip;
- historical seller delay rate;
- historical seller cancellation rate;
- historical product delay rate;
- customer historical delay rate, если считать строго as-of.

Но любые исторические признаки нужно считать только из прошлого относительно purchase timestamp и с учётом label availability. Иначе будет leak.

---

# 3. Лик, который я не заметил

## 3.1 `shipping_limit_max`

**[Высоко]**

Если он выставляется после approval или после фактического принятия заказа в фулфилмент, это post-treatment.

Почему опасен:

- сильно связан с логистическим сроком;
- может быть производной от задержки оплаты;
- может отражать уже возникшие проблемы с заказом;
- если использовать его как признак, модель получит информацию из будущего относительно decision moment.

До доказательства обратного лучше:

```yaml
forbidden:
  - shipping_limit_max
```

или как минимум:

```yaml
ambiguous:
  - shipping_limit_max
```

но не использовать в первом baseline.

---

## 3.2 Payment missingness и payment existence

**[Высоко]**

Если payments table содержит только успешные или зарегистрированные платежи, то наличие/отсутствие платежа — post-treatment.

Признаки типа:

```text
has_payment = 0
payment_value is null
payment_type missing
```

могут кодировать:

- отмену;
- payment failure;
- unavailable order.

Это не известно в момент оформления с точки зрения будущего исхода.

Если используешь payment features, нужно различать:

1. **Выбранный метод и сумма при checkout** — допустимо, если данные фиксируются в момент purchase.
2. **Факт успешного наличия записи платежа** — потенциально post-treatment.

---

## 3.3 Items missingness / zero freight

**[Высоко]**

Если заказы без items чаще являются canceled/unavailable, то признаки:

```text
items_count = 0
items_total = 0
freight_total = 0
```

могут стать прокси для отмены.

Нужно либо:

- исключить такие заказы с отдельным анализом;
- либо честно считать missingness как post-treatment signal и не использовать его как at_decision feature.

---

## 3.4 `order_estimated_delivery_date` как потенциальный post-treatment

**[Средне]**

Сам по себе promised date, скорее всего, доступен при оформлении. Но нужно зафиксировать:

- он назначается до purchase или в момент purchase;
- он не пересчитывается после approval;
- он не является следствием внутренней маршрутизации, произошедшей позже.

Если это не доказано, признак остаётся под вопросом. Но по смыслу задачи promised date должен быть известен покупателю при оформлении.

---

## 3.5 Будущие агрегаты по seller/customer/product

**[Критично для следующих шагов]**

Если позже сделаешь признаки:

```text
seller_late_rate_30d
customer_late_rate_90d
product_delay_rate
category_delay_rate
```

и посчитаешь их по всему датасету без as-of, это будет сильный leak.

Правильно:

```text
для заказа с purchase_time t:
использовать только заказы с purchase_time < t;
учитывать только label_known_at < t, если признак требует исхода.
```

Иначе модель будет знать будущее.

---

# 4. Ошибки, которые проявятся дальше

## 4.1 Split сломается из-за неверного label lag

**[Критично]**

Если embargo 29 дней основан на фактической доставке, а не на моменте узнавания таргета, он может быть:

- слишком большим для label availability;
- слишком маленьким для полного наблюдения late по estimated date;
- неправильно интерпретированным как гарантия наблюдаемости.

Нужно пересчитать:

```text
label_lag = label_known_at - order_purchase_timestamp
```

и затем решить:

- embargo по p95;
- embargo по max;
- или только ретроспективный split без требования label availability на момент cutoff.

Если цель — симуляция продакшена на дату cutoff, нужно использовать только заказы с label_known_at <= cutoff в train.

Если цель — ретроспективное обучение на всём доступном датасете, train labels могут быть известны позже cutoff, но это должно быть осознанным.

---

## 4.2 Test около конца данных может оказаться выборочно «быстрым»

**[Высоко]**

Если взять последний месяц как test, часть заказов может быть unobservable. Если их исключить, test станет смещённым в сторону заказов с короткими обещанными сроками или уже доставленных.

Следствие:

- positive_rate в test может отличаться;
- модель будет оцениваться на нерепрезентативной популяции;
- порог будет выбран неправильно.

Нужно либо:

```text
test_end <= data_end - max_label_lag
```

либо явно принимать неполный test и считать метрики только на наблюдаемых.

---

## 4.3 Baseline должен быть временным

**[Высоко]**

Global positive rate — плохой baseline для out-of-time задачи, если есть дрейф.

Нужны:

- train positive rate;
- validation positive rate;
- test positive rate;
- при необходимости monthly prior baseline.

Метрики:

- accuracy здесь слабо полезна;
- лучше PR-AUC, recall@fixed precision, precision@top-k;
- для калибровки — Brier score, reliability curve.

---

## 4.4 Калибровка и порог пострадают от composite target

**[Высоко]**

Если target включает canceled/unavailable, вероятность, которую калибрует модель, — это вероятность недоставки к дате по любой причине.

Тогда порог для «предупредить о поздней доставке» может быть неверным:

- алерты могут срабатывать на отменённые заказы;
- вмешательство в логистику может быть бесполезно для canceled;
- оценка полезности по историческим late будет смешана.

Перед выбором порога нужно ответить:

```text
Мы оптимизируем предупреждение о логистической задержке?
Или предотвращение любого недоставления к обещанной дате?
```

---

## 4.5 Target encoding и high-cardinality признаки

**[Высоко для модели]**

Если будешь использовать:

- customer_zip_code_prefix;
- customer_city;
- seller_id;
- product_id;
- product category;

нельзя кодировать их target average по всему датасету.

Нужно:

- fit только на train;
- учитывать временной порядок;
- использовать smoothing;
- для seller/customer history — strict as-of.

Иначе появится target leakage.

---

## 4.6 Повторные покупатели

**[Средне]**

`customer_unique_id` имеет повторные заказы. Это не обязательно leak для сырых признаков, но:

- оценки дисперсии могут быть слишком оптимистичными;
- если появятся customer history features, нужен временной as-of;
- при split можно рассмотреть группировку/валидацию по клиентам, если клиентские паттерны важны.

---

# 5. Что в контракте неверно или неполно

## 5.1 `target.definition`

**[Критично]**

Нужно заменить:

```yaml
definition: order_delivered_customer_date > order_estimated_delivery_date
```

на date-only версию:

```yaml
definition: >
  date(order_delivered_customer_date) > date(order_estimated_delivery_date)
```

И отдельно прописать:

- доставка в обещанную дату = on_time;
- как трактуется snapshot day;
- как трактуется missing estimated_date;
- как трактуется delivered status без delivery date.

---

## 5.2 `target.never_delivered_after_estimate`

**[Критично]**

Сейчас:

```yaml
never_delivered_after_estimate: late
```

Это допустимо только для composite outcome. Нужно добавить:

```yaml
estimand: failure_to_deliver_by_promised_date
# или
estimand: logistics_delay_only
```

И указать политику для:

- canceled;
- unavailable;
- invoiced;
- processing;
- shipped.

---

## 5.3 `label_lag_days`

**[Критично]**

Сейчас:

```yaml
label_lag_days:
  median: 10
  p95: 29
```

Это lag фактической доставки, а не label availability.

Нужно:

```yaml
label_known_lag_days:
  median: ...
  p95: ...
  max: ...
  definition: >
    delivered_date if on-time, otherwise estimated_date
```

---

## 5.4 `horizon_days: 209`

**[Высоко]**

Это максимум фактической задержки доставки, не горизонт таргета.

Лучше убрать или переименовать:

```yaml
max_actual_delivery_lag_days: 209
```

А для label availability использовать отдельный horizon.

---

## 5.5 `observation.outcome_observable_for`

**[Высоко]**

Сейчас 0.9999 следует из 4 unobservable. После пересмотра date boundary и snapshot значение может измениться.

Нужно добавить:

```yaml
snapshot_date: ...
snapshot_rule: ...
unobservable_rule: >
  no delivery and estimated_date >= snapshot_date
```

---

## 5.6 `features.at_decision`

**[Высоко]**

Список неполный и местами рискованный.

Нужно добавить или явно решить:

```yaml
- product_weight_g
- product_volume_cm3
- product_photos_qty
- product_category_name
- seller_id
- seller_zip_code_prefix
- seller_state
- payment_types
```

И задать агрегацию:

```yaml
items:
  items_count: count
  items_total: sum(price)
  freight_total: sum(freight_value)
  sellers_count: n_unique(seller_id)
  product_weight_g: sum/max?
  product_volume_cm3: sum/max?
payments:
  payment_value: sum
  payment_installments: max/sum?
  payment_types: set/dominant/one-hot
```

`shipping_limit_max` лучше убрать из at_decision до доказательства.

---

## 5.7 `features.forbidden`

**[Высоко]**

Добавить явно:

```yaml
- review_comment_title
- review_comment_message
- review_creation_date
- review_answer_timestamp
- order_approved_at
- order_delivered_carrier_date
- order_delivered_customer_date
- order_status
- shipping_limit_max  # пока не доказано at_decision
```

И указать, что missingness, порождённая post-treatment событиями, не должна использоваться как обычный at_decision признак.

---

## 5.8 `joins`

**[Средне/Высоко]**

Сейчас есть правильное правило aggregate_before_join, но нет политики для:

- orders without items;
- orders without payments;
- orders without reviews;
- multiple reviews;
- geolocation duplicates;
- zip/city inconsistencies.

Нужно добавить:

```yaml
orphan_policy:
  no_items: exclude_or_flag_or_impute
  no_payments: exclude_or_flag_or_impute
  no_reviews: ignore_if_not_using_reviews
geolocation:
  dedup_strategy: ...
```

---

## 5.9 `split`

**[Критично для следующих шагов]**

Сейчас:

```yaml
split:
  strategy: out_of_time
  embargo_days: 29
```

Недостаточно.

Нужно:

```yaml
split:
  strategy: out_of_time
  time_column: order_purchase_timestamp
  train_end: ...
  validation_start: ...
  validation_end: ...
  test_start: ...
  test_end: ...
  embargo_days: ...  # после пересчёта label_known_lag
  label_availability_policy: >
    use only orders with label_known_at <= observation_end
    or retrospective labels allowed
  unobservable_policy: exclude
```

Embargo должно основываться на corrected label lag, а не на фактической доставке.

---

## 5.10 `action`

**[Средне]**

Если target включает отмены, action нужно уточнить:

```text
Предупредить о возможной недоставке к обещанной дате
```

а не:

```text
Предупредить о поздней доставке
```

Иначе бизнес-интерпретация будет расходиться с целевой переменной.

---

# Итоговый минимальный набор действий перед следующим шагом

1. Переделать таргет на date-only:
   ```text
   late = date(delivered) > date(estimated)
   ```

2. Задать snapshot:
   ```text
   snapshot = max фактических событий наблюдения, не estimated
   ```

3. Пересчитать unobservable с дневной границей:
   ```text
   unobservable = no_delivery AND estimated_date >= snapshot_date
   ```

4. Пересчитать label lag как:
   ```text
   label_known_at = delivered_date if on-time else estimated_date
   ```

5. Проверить состав `late_never_delivered` по `order_status` и решить estimand:
   - composite failure-to-deliver;
   - logistics delay only;
   - multi-target.

6. Перевести `shipping_limit_max` в forbidden/недоступный до доказательства.

7. Разобрать orders without items/payments: не импутировать нули молча.

8. Пересчитать positive_rate, leak-проверки и embargo после всех изменений.