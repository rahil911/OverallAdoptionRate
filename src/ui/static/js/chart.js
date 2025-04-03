/**
 * Overall Adoption Rate Chatbot - Chart Visualization
 * 
 * This script manages the chart visualization functionality including:
 * - Loading and rendering adoption rate data
 * - Handling metric and time period selection
 * - Highlighting anomalies and peaks/valleys
 * - Supporting responsive design
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chartContainer = document.getElementById('adoptionRateChart');
    const timeRangeSelector = document.getElementById('timeRangeSelector');
    const metricSelector = document.getElementById('metricSelector');
    
    // Chart state
    let chartData = {
        adoption_data: [],
        peaks_valleys: [],
        anomalies: [],
        metrics: {
            current: { current: null, average: null },
            changes: { mom_change: null, yoy_change: null }
        }
    };
    
    /**
     * Initialize the chart
     */
    function initChart() {
        // Add event listeners to controls
        timeRangeSelector.addEventListener('change', updateChart);
        metricSelector.addEventListener('change', updateChart);
        
        // Load initial data
        loadChartData();
    }
    
    /**
     * Load chart data from the API
     */
    function loadChartData() {
        const timeRange = timeRangeSelector.value;
        const metricType = metricSelector.value;
        
        console.log(`[DEBUG] Loading chart data: timeRange=${timeRange}, metricType=${metricType}`);
        
        // Show loading state
        showChartLoading();
        
        // Fetch data from API
        console.log(`[DEBUG] Fetching data from API: /api/chart_data?time_range=${timeRange}&metric_type=${metricType}`);
        fetch(`/api/chart_data?time_range=${timeRange}&metric_type=${metricType}`)
            .then(response => {
                console.log(`[DEBUG] API response status: ${response.status} ${response.statusText}`);
                if (!response.ok) {
                    console.error(`API error: ${response.status} ${response.statusText}`);
                    if (response.status === 404) {
                        throw new Error(`No data available for the selected criteria. API returned 404.`);
                    } else {
                        throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
                    }
                }
                return response.json();
            })
            .then(data => {
                console.log('[DEBUG] API returned data:', data);
                console.log(`[DEBUG] Data structure: 
                  - chart_data: ${data.chart_data ? data.chart_data.length + ' items' : 'missing'} 
                  - trends: ${data.trends ? data.trends.length + ' items' : 'missing'}
                  - is_fallback: ${data.is_fallback ? 'true' : 'false'}`);
                
                // Check if the data is empty or undefined
                if (!data || !data.chart_data || data.chart_data.length === 0) {
                    console.error('[DEBUG] Data is empty or missing chart_data array');
                    throw new Error('No data available for the selected criteria');
                }
                
                // Sample of the data
                if (data.chart_data && data.chart_data.length > 0) {
                    console.log('[DEBUG] First data point:', data.chart_data[0]);
                }
                
                // Add fallback indicator if using synthetic data
                if (data.is_fallback) {
                    console.warn('[DEBUG] Using fallback data instead of real database data');
                    addFallbackIndicator();
                } else {
                    removeFallbackIndicator();
                }
                
                // Update chart state
                chartData = data;
                
                // Render chart
                renderChart();
                
                // Hide loading state
                hideChartLoading();
            })
            .catch(error => {
                console.error('[DEBUG] Error loading chart data:', error);
                
                // Show error state with retry button
                showChartError(error.message);
                
                // Hide loading state
                hideChartLoading();
            });
    }
    
    /**
     * Render the chart with current data
     */
    function renderChart() {
        // Extract data
        const chartDataPoints = chartData.chart_data || [];
        const trends = chartData.trends || [];
        
        // Check if we have data
        if (chartDataPoints.length === 0) {
            showChartError('No data available for the selected criteria');
            return;
        }
        
        // Log the data we received to help debug
        console.log('Chart data received:', chartDataPoints);
        console.log('Trends received:', trends);
        
        // Prepare data for main series
        const dates = chartDataPoints.map(item => item.date);
        const values = chartDataPoints.map(item => item.value);
        
        // Main data trace
        const mainTrace = {
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines',
            name: getMetricLabel(),
            line: {
                color: '#2563eb',
                width: 2
            }
        };
        
        // Create traces for peaks and valleys from trends
        let trendTraces = [];
        if (trends && trends.length > 0) {
            const peakPoints = trends
                .filter(item => item.type === 'peak')
                .map(item => ({
                    x: [item.date],
                    y: [item.value],
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Peak',
                    marker: {
                        color: '#10b981',
                        size: 10,
                        symbol: 'circle'
                    },
                    showlegend: false,
                    hoverinfo: 'x+y+text',
                    hovertext: `Peak: ${item.value.toFixed(2)}%`
                }));
            
            const valleyPoints = trends
                .filter(item => item.type === 'valley')
                .map(item => ({
                    x: [item.date],
                    y: [item.value],
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Valley',
                    marker: {
                        color: '#ef4444',
                        size: 10,
                        symbol: 'circle'
                    },
                    showlegend: false,
                    hoverinfo: 'x+y+text',
                    hovertext: `Valley: ${item.value.toFixed(2)}%`
                }));
                
            trendTraces = [...peakPoints, ...valleyPoints];
        }
        
        // Combine all traces
        const traces = [mainTrace, ...trendTraces];
        
        // Chart layout
        const layout = {
            title: {
                text: `${getMetricLabel()} Over Time`,
                font: {
                    family: 'Segoe UI, Roboto, Arial, sans-serif',
                    size: 18
                }
            },
            xaxis: {
                title: 'Date',
                showgrid: true,
                gridcolor: '#f1f5f9'
            },
            yaxis: {
                title: 'Adoption Rate (%)',
                showgrid: true,
                gridcolor: '#f1f5f9',
                ticksuffix: '%'
            },
            margin: {
                l: 50,
                r: 20,
                t: 50,
                b: 50
            },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.2
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        // Responsive config
        const config = {
            responsive: true,
            displayModeBar: false
        };
        
        // Render the chart
        try {
            Plotly.newPlot(chartContainer, traces, layout, config);
        } catch (error) {
            console.error('Error rendering chart:', error);
            showChartError('Error rendering chart: ' + error.message);
        }
    }
    
    /**
     * Get annotations for current metrics
     */
    function getMetricAnnotations() {
        const annotations = [];
        const metrics = chartData.metrics || {};
        
        // Get the latest date
        let lastDate = null;
        const adoption_data = chartData.adoption_data || [];
        if (adoption_data.length > 0) {
            lastDate = adoption_data[adoption_data.length - 1].date;
        }
        
        // Add current value annotation
        if (metrics.current && metrics.current.current !== null) {
            annotations.push({
                x: 1,
                y: 1.05,
                xref: 'paper',
                yref: 'paper',
                text: `Current: ${metrics.current.current}%`,
                showarrow: false,
                font: {
                    family: 'Segoe UI, Roboto, Arial, sans-serif',
                    size: 14,
                    color: '#2563eb'
                },
                align: 'right'
            });
        }
        
        // Add MoM change annotation
        if (metrics.changes && metrics.changes.mom_change !== null) {
            const momChange = metrics.changes.mom_change;
            const changeColor = momChange >= 0 ? '#10b981' : '#ef4444';
            const changeSymbol = momChange >= 0 ? '▲' : '▼';
            
            annotations.push({
                x: 0,
                y: 1.05,
                xref: 'paper',
                yref: 'paper',
                text: `Month-over-Month: ${changeSymbol} ${Math.abs(momChange).toFixed(2)}%`,
                showarrow: false,
                font: {
                    family: 'Segoe UI, Roboto, Arial, sans-serif',
                    size: 14,
                    color: changeColor
                },
                align: 'left'
            });
        }
        
        return annotations;
    }
    
    /**
     * Get the label for the selected metric
     */
    function getMetricLabel() {
        const metricType = metricSelector.value;
        
        switch (metricType) {
            case 'daily':
                return 'Daily Adoption Rate';
            case 'weekly':
                return 'Weekly Adoption Rate';
            case 'monthly':
                return 'Monthly Adoption Rate';
            case 'yearly':
                return 'Yearly Adoption Rate';
            default:
                return 'Adoption Rate';
        }
    }
    
    /**
     * Show loading state for the chart
     */
    function showChartLoading() {
        if (!chartContainer) return;
        
        // Clear existing chart
        chartContainer.innerHTML = '';
        
        // Create loading element
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chart-loading';
        loadingDiv.innerHTML = `
            <div class="chart-loading-spinner"></div>
            <div class="chart-loading-text">Loading chart data...</div>
        `;
        
        // Add to chart container
        chartContainer.appendChild(loadingDiv);
    }
    
    /**
     * Hide loading state
     */
    function hideChartLoading() {
        if (!chartContainer) return;
        
        // Remove loading element if it exists
        const loadingElement = chartContainer.querySelector('.chart-loading');
        if (loadingElement) {
            loadingElement.remove();
        }
    }
    
    /**
     * Show error state in the chart area
     */
    function showChartError(message) {
        if (!chartContainer) return;
        
        // Clear existing chart
        chartContainer.innerHTML = '';
        
        // Create error message element
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chart-error';
        errorDiv.innerHTML = `
            <div class="chart-error-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="chart-error-message">
                ${message}
            </div>
            <button class="chart-retry-button" onclick="updateChart()">
                <i class="fas fa-sync-alt"></i> Retry
            </button>
        `;
        
        // Add to chart container
        chartContainer.appendChild(errorDiv);
    }
    
    /**
     * Update the chart with new data
     */
    function updateChart() {
        console.log('Updating chart...');
        loadChartData();
    }
    
    /**
     * Reset the chart
     */
    function resetChart() {
        // Reset selectors to defaults
        timeRangeSelector.value = '1y';
        metricSelector.value = 'monthly';
        
        // Reload data
        loadChartData();
    }
    
    /**
     * Add fallback data indicator to the chart
     */
    function addFallbackIndicator() {
        // Remove existing indicator first
        removeFallbackIndicator();
        
        // Create fallback indicator
        const indicator = document.createElement('div');
        indicator.className = 'fallback-indicator';
        indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Using demo data (database unavailable)';
        
        // Add to chart container parent
        if (chartContainer && chartContainer.parentNode) {
            chartContainer.parentNode.insertBefore(indicator, chartContainer);
        }
    }
    
    /**
     * Remove fallback data indicator
     */
    function removeFallbackIndicator() {
        const existingIndicator = document.querySelector('.fallback-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
    }
    
    // Initialize the chart
    initChart();
    
    // Export functions
    window.updateChart = updateChart;
    window.resetChart = resetChart;
}); 