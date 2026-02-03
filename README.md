# signalflow-ta

Technical analysis extension for [signalflow-trading](https://pypi.org/project/signalflow-trading/).

## Installation

```bash
pip install signalflow-ta
```

## Requirements

- Python ≥ 3.12
- pandas-ta

## Modules


## Usage

```python
import signalflow.ta as ta
```

## License

See [signalflow-trading](https://pypi.org/project/signalflow-trading/) for license details.


Мені потрібно додати нормалізацію отриманих результатів по кожній з Features
Потрібно додати булевий атрибут який би б визначав, яке значення потрібно повертати
відносне чи абсолютне, також у випадку чітко визначених меж індикатора варто scale значення до [0, 1] [-1, 1] # варто вказати атрибутом класу
Наприклад
RSI - це відносний індикатор, потрібно лише зробити scale (rsi/100)
SmaSmooth - це абсолютний індикатор, який напряму залежить від масштабу ціни. тут ми можемо отримати відсоток різниці у відсотках між поточною ціною та згладженням - це вже відносний індикатор, також можна використовувати rolling z score для приведення до відносного значення
Вибери лише один спосіб приведення до нормального значення.
Також подумай як вірно назвати колонки, щоб в обох режимах вони називались однаково