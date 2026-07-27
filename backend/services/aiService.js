const Transaction = require('../models/Transaction');

/**
 * AI Service for Prediction, Categorization, and Insights
 */
class AIService {
  /**
   * Linear Regression Model: y = mx + c (Ordinary Least Squares)
   * Calculates next month's predicted spending, slope, intercept, and dynamic R² confidence score
   */
  async predictNextMonth(userId) {
    const now = new Date();
    const data = [];

    // Extract last 6 months grouping by month
    for (let i = 5; i >= 0; i--) {
      const start = new Date(now.getFullYear(), now.getMonth() - i, 1);
      const end = new Date(now.getFullYear(), now.getMonth() - i + 1, 0, 23, 59, 59);
      const txns = await Transaction.find({ user: userId, type: 'expense', date: { $gte: start, $lte: end } });
      data.push(txns.reduce((s, t) => s + t.amount, 0));
    }

    // Need at least 3 months with data
    const monthsWithData = data.filter(v => v > 0).length;
    if (monthsWithData < 3) {
      return { success: false, code: 'insufficient_data', message: 'Need at least 3 months of history for a prediction.' };
    }

    // Ordinary Least Squares (OLS) Linear Regression: y = mx + c
    // x = month index (1, 2, 3, 4, 5, 6), y = total monthly expense
    const n = data.length; // 6 time data points
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;

    for (let i = 0; i < n; i++) {
      const x = i + 1; // Month 1 to 6
      const y = data[i];
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
      sumYY += y * y;
    }

    // Slope (m) = (N*Σ(xy) - Σx*Σy) / (N*Σ(x²) - (Σx)²)
    const denominator = (n * sumXX - sumX * sumX);
    const m = denominator !== 0 ? (n * sumXY - sumX * sumY) / denominator : 0;

    // Intercept (c) = (Σy - m*Σx) / N
    const c = (sumY - m * sumX) / n;

    // Predict for Month 7 (x = 7)
    const nextMonthX = 7;
    let prediction = m * nextMonthX + c;
    if (prediction < 0) prediction = 0; // Spending cannot be negative

    // Calculate R² (Coefficient of Determination) for confidence score
    const meanY = sumY / n;
    let ssTot = 0, ssRes = 0;
    for (let i = 0; i < n; i++) {
      const x = i + 1;
      const y = data[i];
      const yPred = m * x + c;
      ssTot += Math.pow(y - meanY, 2);
      ssRes += Math.pow(y - yPred, 2);
    }
    const rSquared = ssTot === 0 ? 1 : Math.max(0, 1 - (ssRes / ssTot));
    const confidence = Math.max(70, Math.min(95, Math.round(rSquared * 100)));

    const trend = m > 0 ? "increasing" : "decreasing";

    return {
      success: true,
      predicted: Math.round(prediction * 100) / 100,
      slope: Math.round(m * 100) / 100,
      intercept: Math.round(c * 100) / 100,
      confidence: confidence,
      historicalData: data,
      trend: trend
    };
  }

  /**
   * Categorization Engine (Pseudo-NLP)
   * Maps description patterns to human-readable categories
   */
  categorizeDescription(description) {
    if (!description) return { category: 'Other', confidence: 0 };
    
    const text = description.toLowerCase().trim();
    const patternMap = [
      { category: 'Food', regex: /(zomato|swiggy|restaurant|food|cafe|groceries|pizza|burger|eat|lunch|coffee)/i, confidence: 0.95 },
      { category: 'Transport', regex: /(uber|ola|taxi|metro|bus|fuel|petrol|diesel|toll|cab|train|travel)/i, confidence: 0.90 },
      { category: 'Shopping', regex: /(amazon|flipkart|shopping|clothes|bought|purchase|mall|cloth)/i, confidence: 0.85 },
      { category: 'Utilities', regex: /(electricity|wifi|rent|bill|gas|water|internet|mobile|house)/i, confidence: 0.90 },
      { category: 'Entertainment', regex: /(movie|netflix|spotify|game|entertainment|show)/i, confidence: 0.80 },
      { category: 'Health', regex: /(doctor|hospital|med|pharm|health|gym|checkup)/i, confidence: 0.95 },
      { category: 'Education', regex: /(course|learn|book|college|school|fee|udemy)/i, confidence: 0.85 },
      { category: 'Investment', regex: /(invest|stock|mutual|sip|fund|crypto)/i, confidence: 0.90 },
      { category: 'Salary', regex: /(salary|stipend|freelance|income|dividend)/i, confidence: 0.95 },
    ];

    for (const item of patternMap) {
      if (item.regex.test(text)) {
        return { category: item.category, confidence: item.confidence };
      }
    }

    return { category: 'Other', confidence: 0.1 };
  }

  /**
   * Deep Insights & Velocity Monitor
   */
  async getInsights(userId) {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    
    // Previous Month Comparison
    const sameDayLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
    const lastMonthDayStart = new Date(sameDayLastMonth.getFullYear(), sameDayLastMonth.getMonth(), sameDayLastMonth.getDate());
    const lastMonthDayEnd = new Date(lastMonthDayStart);
    lastMonthDayEnd.setHours(23, 59, 59, 999);

    const [todayTxns, lastMonthDayTxns, thisMonthTxns] = await Promise.all([
      Transaction.find({ user: userId, type: 'expense', date: { $gte: todayStart } }),
      Transaction.find({ user: userId, type: 'expense', date: { $gte: lastMonthDayStart, $lte: lastMonthDayEnd } }),
      Transaction.find({ user: userId, date: { $gte: new Date(now.getFullYear(), now.getMonth(), 1) } })
    ]);

    const todayTotal = todayTxns.reduce((s, t) => s + t.amount, 0);
    const lastMonthDayTotal = lastMonthDayTxns.reduce((s, t) => s + t.amount, 0);
    
    const alerts = [];

    // Velocity Tiers: Warning vs Critical
    if (lastMonthDayTotal > 0) {
      const velocity = ((todayTotal - lastMonthDayTotal) / lastMonthDayTotal) * 100;
      if (velocity >= 50) {
        alerts.push({ 
          level: 'critical', 
          icon: '🚨', 
          title: 'Critical Spending Spike', 
          message: `Your spending velocity is ${velocity.toFixed(0)}% higher than the same day last month! Huge increase detected.` 
        });
      } else if (velocity >= 20) {
        alerts.push({ 
          level: 'warning', 
          icon: '⚠️', 
          title: 'Spending Velocity Warning', 
          message: `You've spent ${velocity.toFixed(0)}% more today than you did on this exact day last month.` 
        });
      }
    }

    // Top Category & Trends
    const catMap = {};
    thisMonthTxns.filter(t => t.type === 'expense').forEach(t => {
      catMap[t.category] = (catMap[t.category] || 0) + t.amount;
    });
    const sortedCats = Object.entries(catMap).sort((a, b) => b[1] - a[1]);
    const topCategory = sortedCats.length > 0 ? { name: sortedCats[0][0], amount: sortedCats[0][1] } : null;

    // Monthly Savings Rate (Simulated trend)
    const totalExp = thisMonthTxns.filter(t => t.type === 'expense').reduce((s, t) => s + t.amount, 0);
    const totalInc = thisMonthTxns.filter(t => t.type === 'income').reduce((s, t) => s + t.amount, 0);
    const savingsPercent = totalInc > 0 ? ((totalInc - totalExp) / totalInc) * 100 : 0;

    return {
      alerts,
      topCategory,
      savingsPercent: Math.round(savingsPercent),
      todayTotal,
      lastMonthDayTotal
    };
  }
}

module.exports = new AIService();
