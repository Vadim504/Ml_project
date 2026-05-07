// public_html/js/backtest.js

// Глобальные переменные
let btChart = null;
let btAbortController = null;

// ============================================================================
// 🚀 ОСНОВНАЯ ФУНКЦИЯ: ЗАПУСК БЭКТЕСТА
// ============================================================================

async function runBacktest() {
    // 1. Считываем и валидируем параметры
    const symbol = document.getElementById('bt-symbol')?.value?.toUpperCase()?.trim();
    const model = document.getElementById('bt-model')?.value;
    const trainDays = parseInt(document.getElementById('bt-train')?.value || '30');
    const testDays = parseInt(document.getElementById('bt-test')?.value || '10');

    // Валидация входных данных
    if (!symbol || symbol.length < 1) {
        alert('⚠️ Пожалуйста, введите тикер акции');
        return;
    }
    
    if (!['cnn', 'lstm', 'gru', 'default'].includes(model)) {
        alert('⚠️ Неверный тип модели');
        return;
    }
    
    if (trainDays < 10 || trainDays > 500) {
        alert('⚠️ Период обучения: от 10 до 500 дней');
        return;
    }
    
    if (testDays < 5 || testDays > 100) {
        alert('⚠️ Период теста: от 5 до 100 дней');
        return;
    }
    
    if (trainDays + testDays > 600) {
        alert('⚠️ Сумма train + test не должна превышать 600 дней');
        return;
    }

    // 2. UI: блокируем кнопку, показываем загрузку
    const btn = document.getElementById('bt-run-btn');
    const resultsCard = document.getElementById('results-card');
    const loadingDiv = document.getElementById('bt-loading');
    
    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Тестирование...';
    }
    
    if (resultsCard) resultsCard.style.display = 'none';
    if (loadingDiv) loadingDiv.style.display = 'block';

    // 3. Отменяем предыдущий запрос если есть
    if (btAbortController) {
        btAbortController.abort();
    }
    btAbortController = new AbortController();

    try {
        // 4. ✅ Безопасная сборка параметров (защита от спецсимволов)
        const params = new URLSearchParams({
            symbol: symbol,
            model: model,
            train_days: trainDays.toString(),
            test_days: testDays.toString()
        });

        // 5. Запрос к API
        const response = await fetch(`/api/backtest?${params}`, {
            signal: btAbortController.signal,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
        
        // 6. Обработка ошибок HTTP
        if (!response.ok) {
            let errorMsg = `HTTP ошибка ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.detail || errorData.error || errorMsg;
            } catch (e) {
                // Если не удалось распарсить JSON, берём текст
                errorMsg = await response.text() || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        // 7. Парсинг ответа
        const data = await response.json();
        
        // 8. Проверка на бизнес-ошибки в ответе
        if (data.error || data.detail) {
            throw new Error(data.detail || data.error);
        }
        
        // 9. Отображение результатов
        displayBacktestResults(data);
        
    } catch (err) {
        // Обработка отмены запроса
        if (err.name === 'AbortError') {
            console.log('🔄 Запрос отменён пользователем');
            return;
        }
        
        // Обработка остальных ошибок
        console.error('❌ Ошибка backtest:', err);
        alert('❌ Ошибка: ' + err.message);
        
        // Показываем сообщение об ошибке в UI
        const resultsCard = document.getElementById('results-card');
        if (resultsCard) {
            resultsCard.style.display = 'block';
            const tbody = document.getElementById('bt-transactions');
            if (tbody) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:#dc3545">
                    ⚠️ ${err.message}
                </td></tr>`;
            }
        }
        
    } finally {
        // 10. Восстановление UI
        const btn = document.getElementById('bt-run-btn');
        const loadingDiv = document.getElementById('bt-loading');
        
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🚀 Запустить Backtest';
        }
        if (loadingDiv) loadingDiv.style.display = 'none';
    }
}


// ============================================================================
// 📊 ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
// ============================================================================

function displayBacktestResults(data) {
    console.log("✅ Бэктест завершен. Получены данные:", data);

    // 1. Показываем контейнеры
    const chartsContainer = document.getElementById('chartsContainer');
    if (chartsContainer) chartsContainer.classList.remove('hidden');

    // 2. Отрисовка карточек (если функция называется renderResults)
    if (typeof renderResults === 'function' && data.results) {
        renderResults(data.results);
    }

    // 3. Отрисовка графиков
    if (typeof renderCharts === 'function' && data.chart_data) {
        renderCharts(data.chart_data);
    }

    // 4. Отрисовка ТАБЛИЦЫ СРАВНЕНИЯ (САМОЕ ВАЖНОЕ)
    if (data.results) {
        const tableBody = document.getElementById('metricsTableBody');
        if (tableBody) {
            tableBody.innerHTML = ''; // Очистка
            
            Object.entries(data.results).forEach(([name, res]) => {
                const row = document.createElement('tr');
                row.className = "border-b last:border-0 hover:bg-gray-50 transition-colors";
                
                // Безопасно вынимаем метрики (ставим 0 если их нет)
                const wr = res.win_rate || 0;
                const mdd = res.max_dd || 0;
                const apnl = res.avg_pnl || 0;
                const sh = res.sharpe || 0;

                row.innerHTML = `
                    <td class="p-4 text-blue-600 font-black">${name.toUpperCase()}</td>
                    <td class="p-4">${wr}%</td>
                    <td class="p-4 text-red-500">${mdd}%</td>
                    <td class="p-4 ${apnl >= 0 ? 'text-green-500' : 'text-red-500'}">${apnl}%</td>
                    <td class="p-4 text-purple-600">${sh}</td>
                `;
                tableBody.appendChild(row);
            });
            console.log("📊 Таблица сравнения заполнена");
        } else {
            console.error("❌ Ошибка: не найден элемент #metricsTableBody");
        }
    }
}
// Вспомогательная функция для установки значения метрики
function setMetricValue(elementId, value, isPositive) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = value;
        el.className = `metric-value ${isPositive ? 'positive' : 'negative'}`;
    }
}

// Форматирование валюты
function formatCurrency(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${Math.abs(value).toFixed(2)}`;
}

// ============================================================================
// 🏆 ОТРИСОВКА ТАБЛИЦЫ СРАВНЕНИЯ МЕТРИК
// ============================================================================

// ============================================================================
// 📈 ОТРИСОВКА ГРАФИКА (Chart.js)
// ============================================================================

function renderBacktestChart(data) {
    const canvas = document.getElementById('bt-chart');
    if (!canvas) return;
    
    // ✅ Проверка: загружен ли Chart.js
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js не загружен. Проверьте подключение скрипта.');
        return;
    }
    
    const portfolioValues = data.portfolio_values || [];
    const testPrices = data.test_prices || [];
    
    if (portfolioValues.length === 0) {
        console.warn('⚠️ Нет данных для графика');
        return;
    }
    
    // Уничтожаем старый график если есть
    if (btChart) {
        btChart.destroy();
        btChart = null;
    }
    
    const ctx = canvas.getContext('2d');
    const labels = portfolioValues.map((_, i) => `День ${i + 1}`);
    
    btChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Цена акции ($)',
                    data: testPrices,
                    borderColor: '#4bc0c0',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    yAxisID: 'y',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'Стоимость портфеля ($)',
                    data: portfolioValues,
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.15)',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    yAxisID: 'y1',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 750,
                easing: 'easeOutQuart'
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: { size: 14, weight: 'bold' },
                    bodyFont: { size: 13 },
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Цена акции ($)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Портфель ($)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        drawOnChartArea: false,
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'День тестового периода'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.03)'
                    }
                }
            }
        }
    });
}


// ============================================================================
// 📋 ОТРИСОВКА ТАБЛИЦЫ СДЕЛОК
// ============================================================================

function renderTransactions(results) {
    const tbody = document.getElementById('bt-transactions');
    if (!tbody) return;

    // Фильтруем только реальные сделки (исключаем hold)
    const trades = (results || []).filter(r => r.action && r.action !== 'hold');
    
    if (trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align:center;padding:20px;color:#666">
                    📭 Нет сделок за тестовый период
                </td>
            </tr>
        `;
        return;
    }
    
    // Показываем последние 20 сделок в обратном порядке (сначала новые)
    const recentTrades = trades.slice(-20).reverse();
    
    tbody.innerHTML = recentTrades.map(r => {
        const signalClass = `signal-${r.signal?.toLowerCase() || 'hold'}`;
        const actionText = r.action?.toUpperCase() || 'HOLD';
        const signalText = r.signal?.toUpperCase() || 'HOLD';
        const price = r.price ? `$${r.price.toFixed(2)}` : '-';
        const balance = r.balance !== undefined ? `$${r.balance.toFixed(2)}` : '-';
        const profit = r.profit !== undefined 
            ? `<span class="${r.profit >= 0 ? 'positive' : 'negative'}">
                ${r.profit >= 0 ? '+' : ''}$${r.profit.toFixed(2)}
               </span>` 
            : '-';
        
        return `
            <tr>
                <td><strong>${r.day || '-'}</strong></td>
                <td>${r.date || '-'}</td>
                <td class="${signalClass}">${signalText}</td>
                <td>${actionText}</td>
                <td>${price}</td>
                <td>
                    ${balance}<br>
                    <small style="color:#666">${profit}</small>
                </td>
            </tr>
        `;
    }).join('');
}


// ============================================================================
// 🗑️ ОЧИСТКА РЕЗУЛЬТАТОВ
// ============================================================================

function clearBacktestResults() {
    // Скрываем карточку результатов
    const resultsCard = document.getElementById('results-card');
    if (resultsCard) {
        resultsCard.style.display = 'none';
    }
    
    // Уничтожаем график
    if (btChart) {
        btChart.destroy();
        btChart = null;
    }
    
    // Очищаем таблицу
    const tbody = document.getElementById('bt-transactions');
    if (tbody) {
        tbody.innerHTML = '';
    }
    
    // Сбрасываем метрики
    ['bt-accuracy', 'bt-profit', 'bt-final', 'bt-drawdown', 'bt-trades', 'bt-winrate'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = '-';
            el.className = 'metric-value';
        }
    });
    
    console.log('🗑️ Результаты очищены');
}


// ============================================================================
// 🎯 ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ СТРАНИЦЫ
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ backtest.js загружен');
    
    // Проверяем наличие критических элементов
    const required = ['bt-symbol', 'bt-model', 'bt-run-btn', 'results-card'];
    required.forEach(id => {
        if (!document.getElementById(id)) {
            console.warn(`⚠️ Элемент #${id} не найден в DOM`);
        }
    });
    
    // Проверяем Chart.js
    if (typeof Chart === 'undefined') {
        console.warn('⚠️ Chart.js не загружен. Проверьте подключение CDN в HTML.');
    }
    
    // Добавляем обработчик нажатия Enter в поле тикера
    const symbolInput = document.getElementById('bt-symbol');
    if (symbolInput) {
        symbolInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                runBacktest();
            }
        });
    }
    
    // Кнопка "Очистить" (если есть в HTML)
    const clearBtn = document.getElementById('bt-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearBacktestResults);
    }
});


// ============================================================================
// 🧪 УТИЛИТЫ ДЛЯ ОТЛАДКИ (только в dev)
// ============================================================================

// Вызовите в консоли: debugBacktest()
function debugBacktest() {
    console.log('🔍 Debug info:');
    console.log('  - Chart.js:', typeof Chart !== 'undefined' ? '✅' : '❌');
    console.log('  - DOM loaded:', document.readyState);
    console.log('  - Elements:', {
        symbol: !!document.getElementById('bt-symbol'),
        runBtn: !!document.getElementById('bt-run-btn'),
        results: !!document.getElementById('results-card'),
        chart: !!document.getElementById('bt-chart')
    });
}

// Экспорт функций для использования из консоли
window.runBacktest = runBacktest;
window.clearBacktestResults = clearBacktestResults;
window.debugBacktest = debugBacktest;