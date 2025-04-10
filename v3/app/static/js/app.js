/**
 * Recommendation System v2 - JavaScript Frontend
 * Handles user interactions with the enhanced recommendation system
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing application...');
    
    // Initialize tabs
    initTabs();
    
    // Load data
    loadUsers();
    loadItems();
    
    // Set up sliders for hybrid recommendations
    initWeightSliders();
    
    // Initialize demographic factors
    initDemographicFactors();
    
    // Set up event listeners
    setupEventListeners();

    // NCF tab event listeners
    document.getElementById('get-ncf-recommendations').addEventListener('click', getNcfRecommendations);
    document.getElementById('train-ncf-model').addEventListener('click', trainNcfModel);
    
    // Populate user select in NCF tab using the correct function
    populateUserDropdown('ncf-user-select', users);
});

// Global variables to store users and items
let users = [];
let items = [];

// Initialize the application
function initApp() {
    console.log('Initializing recommendation system v2');
    
    // Load users and items for dropdowns
    loadUsers();
    loadItems();
    
    // Initialize tab functionality
    initTabs();
    
    // Initialize weight sliders
    initWeightSliders();
    
    // Set up event listeners for forms and buttons
    setupEventListeners();
    
    // Set up event handlers
    setupEventHandlers();
    
    // Log initialization completed
    console.log('Recommendation system v2 initialized successfully');
}

// Set up event handlers
function setupEventHandlers() {
    // User-based recommendation button
    document.getElementById('get-user-recommendations').addEventListener('click', getUserRecommendations);
    
    // Item-based recommendation button
    document.getElementById('get-item-recommendations').addEventListener('click', getItemRecommendations);
    
    // Hybrid recommendation button
    document.getElementById('get-hybrid-recommendations').addEventListener('click', getHybridRecommendations);
    
    // Content-based recommendation button
    document.getElementById('get-content-recommendations').addEventListener('click', getContentRecommendations);
    
    // Demographic recommendation button
    document.getElementById('get-demographic-recommendations').addEventListener('click', getDemographicRecommendations);
}

// Load users into all user dropdowns
function loadUsers() {
    console.log('Loading users...');
    fetch('/api/users')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load users: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            users = data;
            console.log(`Loaded ${users.length} users`);
            
            // Populate all user dropdowns
            populateUserDropdown('user-select', users);
            populateUserDropdown('hybrid-user-select', users);
            populateUserDropdown('demographic-user-select', users);
            populateUserDropdown('ncf-user-select', users);
        })
        .catch(error => {
            console.error('Error loading users:', error);
            alert('Failed to load users. Please refresh the page and try again.');
        });
}

// Load items into all item dropdowns
function loadItems() {
    console.log('Loading items...');
    fetch('/api/items')
        .then(response => response.json())
        .then(data => {
            items = data;
            console.log(`Loaded ${items.length} items`);
            populateItemDropdown('item-select', items);
            populateItemDropdown('content-item-select', items);
        })
        .catch(error => console.error('Error loading items:', error));
}

// Populate a user dropdown with user options
function populateUserDropdown(dropdownId, users) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) {
        console.error(`Dropdown with ID ${dropdownId} not found`);
        return;
    }
    
    console.log(`Populating dropdown ${dropdownId} with ${users.length} users`);
    
    // Clear existing options
    dropdown.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select a user...';
    dropdown.appendChild(defaultOption);
    
    // Add user options
    if (users && users.length > 0) {
        // Sort users by ID for better readability
        const sortedUsers = [...users].sort((a, b) => a.user_id - b.user_id);
        
        sortedUsers.forEach(user => {
            const option = document.createElement('option');
            option.value = user.user_id;
            option.textContent = `User ${user.user_id} (${user.gender}, ${user.age_group})`;
            dropdown.appendChild(option);
        });
        
        console.log(`Added ${users.length} users to dropdown ${dropdownId}`);
    } else {
        console.warn(`No users available to populate dropdown ${dropdownId}`);
    }
    
    // Add change handler to show user details
    dropdown.addEventListener('change', function() {
        const userId = this.value;
        if (userId) {
            fetchUserDetails(userId, dropdownId);
        } else {
            // Hide details panel if no user is selected
            const panelId = dropdownId.replace('select', 'details');
            document.getElementById(panelId).style.display = 'none';
        }
    });
}

// Populate an item dropdown with item options
function populateItemDropdown(dropdownId, items) {
    const dropdown = document.getElementById(dropdownId);
    
    // Clear existing options except the first one
    const firstOption = dropdown.options[0];
    dropdown.innerHTML = '';
    dropdown.appendChild(firstOption);
    
    // Add item options
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item.item_id;
        option.textContent = `${item.name} (${item.main_category})`;
        dropdown.appendChild(option);
    });
    
    // Add change handler to show item details
    dropdown.addEventListener('change', function() {
        const itemId = this.value;
        if (itemId) {
            fetchItemDetails(itemId, dropdownId);
        } else {
            // Hide details panel if no item is selected
            const panelId = dropdownId.replace('select', 'details');
            document.getElementById(panelId).style.display = 'none';
        }
    });
}

// Initialize tabs functionality
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Hide all tab contents
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Show the corresponding tab content
            const tabId = this.getAttribute('data-tab');
            const activeContent = document.getElementById(tabId);
            if (activeContent) {
                activeContent.classList.add('active');
                
                // Clear the recommendations when switching tabs
                clearRecommendations();
            }
        });
    });
}

// Clear the recommendations container
function clearRecommendations() {
    const container = document.getElementById('recommendations');
    if (container) {
        container.innerHTML = '<div class="no-results">Select a recommendation method and click the corresponding button to see recommendations</div>';
    }
}

// Initialize weight sliders for hybrid recommendations
function initWeightSliders() {
    const sliders = ['collab-weight', 'content-weight', 'demographic-weight'];
    
    sliders.forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(`${sliderId}-value`);
        
        if (slider && valueDisplay) {
            // Initialize with the starting value
            valueDisplay.textContent = slider.value;
            
            // Update when slider changes
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        }
    });
}

// Fetch user details from the API
function fetchUserDetails(userId, dropdownId) {
    console.log(`Fetching details for user ${userId} (dropdown: ${dropdownId})`);
    
    fetch(`/api/users/${userId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to fetch user details: ${response.status}`);
            }
            return response.json();
        })
        .then(user => {
            console.log(`Received user details for ${userId}:`, user);
            
            // Extract the base ID correctly based on the dropdown ID pattern
            let baseId;
            if (dropdownId === 'user-select') {
                baseId = 'user';
            } else {
                // For hybrid-user-select and demographic-user-select, get the first part
                baseId = dropdownId.split('-')[0];
            }
            
            console.log(`Using baseId: ${baseId} for dropdown: ${dropdownId}`);
            
            // Get the details panel
            const detailsPanel = document.getElementById(`${baseId}-user-details`) || 
                               document.getElementById(`${baseId}-details`);
            
            // Get the financial insights panel and info div
            let financialPanel, financialInfo;
            if (baseId === 'user') {
                financialPanel = document.getElementById('financial-insights');
                financialInfo = document.getElementById('financial-info');
            } else {
                financialPanel = document.getElementById(`${baseId}-financial-insights`);
                financialInfo = document.getElementById(`${baseId}-financial-info`);
            }
            
            console.log(`Panels found:`, {
                detailsPanel: detailsPanel?.id,
                financialPanel: financialPanel?.id,
                financialInfo: financialInfo?.id
            });
            
            // Update user details panel
            if (detailsPanel) {
                const userInfo = detailsPanel.querySelector('div');
                if (userInfo) {
                    userInfo.innerHTML = `
                        <div class="info-row"><span>ID:</span> ${user.user_id}</div>
                        <div class="info-row"><span>Age Group:</span> ${user.age_group}</div>
                        <div class="info-row"><span>Gender:</span> ${user.gender}</div>
                        <div class="info-row"><span>Location:</span> ${user.city}, ${user.country}</div>
                        <div class="info-row"><span>Income:</span> ${user.income_bracket}</div>
                        <div class="info-row"><span>Interests:</span> ${Array.isArray(user.interests) ? user.interests.join(', ') : user.interests}</div>
                        <div class="info-row"><span>Last Active:</span> ${user.last_active}</div>
                    `;
                    detailsPanel.style.display = 'block';
                }
            }
            
            // Now fetch financial insights
            if (financialPanel && financialInfo) {
                console.log(`Fetching financial insights for panel: ${financialPanel.id}`);
                return fetchFinancialInsights(userId, financialInfo).then(() => {
                    financialPanel.style.display = 'block';
                });
            } else {
                console.warn(`Financial panel or info div not found for ${baseId}`, {
                    panelId: financialPanel?.id,
                    infoId: financialInfo?.id
                });
            }
        })
        .catch(error => {
            console.error('Error fetching user details:', error);
            const baseId = dropdownId.split('-')[0];
            const detailsPanel = document.getElementById(`${baseId}-user-details`) || 
                               document.getElementById(`${baseId}-details`);
            if (detailsPanel) {
                const userInfo = detailsPanel.querySelector('div');
                if (userInfo) {
                    userInfo.innerHTML = `<p class="error">Error loading user details: ${error.message}</p>`;
                    detailsPanel.style.display = 'block';
                }
            }
        });
}

// Fetch financial insights for a user
function fetchFinancialInsights(userId, containerId) {
    // Get the container element, whether passed as element or ID
    const container = typeof containerId === 'string' ? 
        document.getElementById(containerId) : 
        containerId;

    if (!container) {
        console.error('Financial insights container not found');
        return Promise.resolve();
    }
    
    console.log(`Fetching financial insights for user ${userId} in container ${container.id}`);
    
    // Show loading indicator
    container.innerHTML = '<p class="loading">Loading financial insights...</p>';
    
    // Find and show the parent container
    const parentContainer = container.parentElement;
    if (parentContainer && parentContainer.classList.contains('financial-insights')) {
        parentContainer.style.display = 'block';
    }
    
    return fetch(`/api/financial/profile/${userId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to fetch financial profile: ${response.status}`);
            }
            return response.json();
        })
        .then(profile => {
            console.log(`Received financial profile for ${userId}:`, profile);
            
            // Ensure default values if profile properties are missing
            const spendingLevel = profile.spending_level || 'Moderate';
            const riskTolerance = profile.risk_tolerance || 'Medium';
            const budgetStatus = profile.budget_status || 'Balanced';
            
            // Format financial insights
            container.innerHTML = `
                <div class="info-row"><span>Spending Level:</span> <strong>${spendingLevel}</strong></div>
                <div class="info-row"><span>Risk Tolerance:</span> <strong>${riskTolerance}</strong></div>
                <div class="info-row"><span>Budget Status:</span> <strong>${budgetStatus}</strong></div>
            `;
            
            // Now fetch financial advice if available
            return fetch(`/api/financial/advice/${userId}`);
        })
        .then(response => {
            if (!response.ok) {
                // It's okay if advice isn't available
                return null;
            }
            return response.json();
        })
        .then(advice => {
            if (advice && advice.warnings && advice.warnings.length > 0) {
                console.log(`Received financial advice for ${userId}:`, advice);
                
                // Add financial advice to the container
                const adviceElement = document.createElement('div');
                adviceElement.className = 'financial-advice warning';
                adviceElement.innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>${advice.warnings[0].message}</span>
                `;
                container.appendChild(adviceElement);
            }
        })
        .catch(error => {
            console.error('Error fetching financial insights:', error);
            container.innerHTML = `<p class="error">Error loading financial insights: ${error.message}</p>`;
            
            // Still keep the container visible even if there's an error
            const parentContainer = container.parentElement;
            if (parentContainer && parentContainer.classList.contains('financial-insights')) {
                parentContainer.style.display = 'block';
            }
        });
}

// Fetch item details for the details panel
function fetchItemDetails(itemId, dropdownId) {
    console.log(`Fetching details for item ${itemId} for dropdown ${dropdownId}`);
    
    fetch(`/api/items/${itemId}`)
        .then(response => {
            console.log(`Item details response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch item: ${response.status}`);
            }
            return response.json();
        })
        .then(item => {
            // Determine which details panel to use based on the dropdown ID
            let panelId;
            if (dropdownId === 'item-select') {
                panelId = 'item-details';
            } else if (dropdownId === 'content-item-select') {
                panelId = 'content-item-details';
            }
            
            // Display item details in the panel
            if (panelId) {
                const panel = document.getElementById(panelId);
                if (!panel) {
                    console.error(`Panel with ID ${panelId} not found`);
                    return;
                }
                
                const infoDiv = panel.querySelector('.item-info') || panel.querySelector('div');
                if (!infoDiv) {
                    console.error(`Info div not found in panel ${panelId}`);
                    return;
                }
                
                const tagsDisplay = Array.isArray(item.tags) ? item.tags.join(', ') : item.tags;
                
                infoDiv.innerHTML = `
                    <div class="info-row"><span>ID:</span> ${item.item_id}</div>
                    <div class="info-row"><span>Name:</span> ${item.name}</div>
                    <div class="info-row"><span>Category:</span> ${item.main_category} > ${item.subcategory}</div>
                    <div class="info-row"><span>Brand:</span> ${item.brand}</div>
                    <div class="info-row"><span>Tags:</span> ${tagsDisplay}</div>
                    <div class="info-row"><span>Price:</span> $${item.price}</div>
                    <div class="info-row"><span>Rating:</span> ${item.average_rating.toFixed(1)} (${item.num_ratings})</div>
                    <div class="info-row"><span>Stock:</span> ${item.stock_level}</div>
                    <div class="info-row"><span>Description:</span> ${item.description}</div>
                `;
                
                panel.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error fetching item details:', error);
            alert('Could not fetch item details. Please try again.');
        });
}

// Show financial insights based on user profile
function showFinancialInsights(user) {
    const financialAwareness = document.getElementById('financial-awareness').value;
    const insightsPanel = document.getElementById('financial-insights');
    const infoDiv = insightsPanel.querySelector('.financial-info');
    
    if (financialAwareness === 'none') {
        insightsPanel.style.display = 'none';
        return;
    }
    
    // Create financial insights based on user profile and selected awareness level
    let insights = '';
    
    if (financialAwareness === 'basic') {
        insights = `
            <p>Based on your income bracket (${user.income_bracket}), we'll highlight recommendations that:</p>
            <ul>
                <li>Fit within your estimated budget</li>
                <li>Provide good value for money</li>
                <li>Include occasional splurges that align with your interests</li>
            </ul>
        `;
    } else if (financialAwareness === 'strict') {
        insights = `
            <p>With strict budget controls for income bracket (${user.income_bracket}), we'll:</p>
            <ul>
                <li>Prioritize essential items within your budget range</li>
                <li>Flag luxury items that may impact your financial well-being</li>
                <li>Suggest alternatives when items exceed your typical spending range</li>
                <li>Apply a 24-hour consideration period for high-value purchases</li>
            </ul>
        `;
    }
    
    infoDiv.innerHTML = insights;
    insightsPanel.style.display = 'block';
}

// Display recommendations in the results container
function displayRecommendations(containerId, recommendations, financialAwareness = 'moderate') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container with id ${containerId} not found`);
        return;
    }
    
    // Clear the container
    container.innerHTML = '';
    
    // Determine which tab is active to set the title
    const activeTab = document.querySelector('.tab.active');
    let recommendationType = "Recommendations";
    
    if (activeTab) {
        const tabType = activeTab.getAttribute('data-tab');
        switch (tabType) {
            case 'user-based':
                recommendationType = "User-Based Recommendations";
                break;
            case 'item-based':
                recommendationType = "Item-Based Recommendations";
                break;
            case 'hybrid':
                recommendationType = "Hybrid Recommendations";
                break;
            case 'content-based':
                recommendationType = "Content-Based Recommendations";
                break;
            case 'demographic':
                recommendationType = "Demographic Recommendations";
                break;
        }
    }
    
    // Add a title to the container
    const titleElement = document.createElement('h2');
    titleElement.className = 'section-title';
    titleElement.textContent = recommendationType;
    container.appendChild(titleElement);
    
    // Add financial awareness level info
    const awarenessInfo = document.createElement('div');
    awarenessInfo.className = 'financial-awareness-info';
    awarenessInfo.innerHTML = `<p><i class="fas fa-info-circle"></i> Financial Awareness Level: <strong>${financialAwareness}</strong></p>`;
    container.appendChild(awarenessInfo);
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML += '<p class="no-results">No recommendations found</p>';
        return;
    }
    
    // Create a div for the results grid
    const resultsGrid = document.createElement('div');
    resultsGrid.className = 'results-grid';
    container.appendChild(resultsGrid);
    
    recommendations.forEach(item => {
        // Create the item card
        const card = document.createElement('div');
        card.className = 'item-card';
        
        // Add financial warning class if needed based on awareness level
        if (financialAwareness === 'low' && item.price > 100) {
            card.classList.add('luxury-item');
        }
        
        // Format tags properly
        let tags = '';
        if (item.tags) {
            // Handle both string and array formats for tags
            if (typeof item.tags === 'string') {
                tags = item.tags.split(',').map(tag => 
                    `<span class="tag">${tag.trim()}</span>`).join('');
            } else if (Array.isArray(item.tags)) {
                tags = item.tags.map(tag => 
                    `<span class="tag">${tag.trim()}</span>`).join('');
            }
        }
        
        // Create the card content - restructured for proper badge display
        card.innerHTML = `
            <div class="item-header">
                <span class="category-badge">${item.main_category || 'Unknown'}</span>
                <span class="subcategory-badge">${item.subcategory || 'General'}</span>
                <span class="score-badge">${item.score ? item.score.toFixed(2) : 'N/A'}</span>
            </div>
            <h4 class="item-title">${item.name}</h4>
            <div class="item-meta">
                <div class="item-brand">${item.brand || ''}</div>
                <div class="item-price ${item.price > 100 ? 'high-price' : ''}">${item.price ? '$' + item.price.toFixed(2) : '$0.00'}</div>
                <div class="item-rating">★ ${item.average_rating ? item.average_rating.toFixed(1) : '0.0'} (${item.num_ratings || 0})</div>
                <div class="item-stock">${getStockLabel(item.stock_level || 0)}</div>
            </div>
            <div class="item-tags">
                ${tags}
            </div>
            <div class="item-description">${truncateText(item.description || '', 100)}</div>
        `;
        
        // Add financial advice based on awareness level and item price
        if (financialAwareness === 'low' && item.price > 100) {
            const advice = document.createElement('div');
            advice.className = 'financial-advice warning';
            advice.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <span>Luxury Item Warning: This item is in the higher price range for your income bracket.</span>
            `;
            card.appendChild(advice);
        } else if (financialAwareness === 'moderate' && item.price > 80) {
            const advice = document.createElement('div');
            advice.className = 'financial-advice moderate';
            advice.innerHTML = `
                <i class="fas fa-info-circle"></i>
                <span>Consider if this purchase aligns with your budget priorities.</span>
            `;
            card.appendChild(advice);
        } else if (financialAwareness === 'low' && item.price > 50) {
            const advice = document.createElement('div');
            advice.className = 'financial-advice info';
            advice.innerHTML = `
                <i class="fas fa-piggy-bank"></i>
                <span>This item is within your budget range.</span>
            `;
            card.appendChild(advice);
        }
        
        resultsGrid.appendChild(card);
    });
    
    // Add a scroll-to behavior to make sure user sees the recommendations
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Helper function to get stock label
function getStockLabel(stockLevel) {
    if (stockLevel <= 0) {
        return '<span class="item-stock out-of-stock">Out of Stock</span>';
    } else if (stockLevel < 5) {
        return '<span class="item-stock low-stock">Low Stock</span>';
    } else {
        return '<span class="item-stock in-stock">In Stock</span>';
    }
}

// Helper function to truncate text
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Set up event listeners
function setupEventListeners() {
    // User-based recommendations
    const getUserRecsBtn = document.getElementById('get-user-recommendations');
    if (getUserRecsBtn) {
        getUserRecsBtn.addEventListener('click', getUserRecommendations);
    }
    
    // Item-based recommendations
    const getItemRecsBtn = document.getElementById('get-item-recommendations');
    if (getItemRecsBtn) {
        getItemRecsBtn.addEventListener('click', getItemRecommendations);
    }
    
    // Hybrid recommendations
    const getHybridRecsBtn = document.getElementById('get-hybrid-recommendations');
    if (getHybridRecsBtn) {
        getHybridRecsBtn.addEventListener('click', getHybridRecommendations);
    }
    
    // Content-based recommendations
    const getContentRecsBtn = document.getElementById('get-content-recommendations');
    if (getContentRecsBtn) {
        getContentRecsBtn.addEventListener('click', getContentRecommendations);
    }
    
    // Demographic recommendations
    const getDemoRecsBtn = document.getElementById('get-demographic-recommendations');
    if (getDemoRecsBtn) {
        getDemoRecsBtn.addEventListener('click', getDemographicRecommendations);
    }
    
    // User selection events
    setupUserSelectionEvents();
    
    // Item selection events
    setupItemSelectionEvents();
    
    console.log('Event listeners set up successfully');
}

// Set up user selection events for all tabs
function setupUserSelectionEvents() {
    const userSelectors = [
        'user-select', 
        'hybrid-user-select', 
        'demographic-user-select',
        'ncf-user-select'  // Added NCF user selector
    ];
    
    userSelectors.forEach(selectorId => {
        const selector = document.getElementById(selectorId);
        if (selector) {
            selector.addEventListener('change', function() {
                const userId = this.value;
                if (userId) {
                    fetchUserDetails(userId, selectorId);
                    // Also fetch financial insights for the user
                    getFinancialInsights(userId, selectorId.replace('select', 'financial-info'));
                } else {
                    // Hide details panel if no user is selected
                    const detailsPanel = document.getElementById(selectorId.replace('select', 'details'));
                    const financialPanel = document.getElementById(selectorId.replace('select', 'financial-insights'));
                    if (detailsPanel) detailsPanel.style.display = 'none';
                    if (financialPanel) financialPanel.style.display = 'none';
                }
            });
        }
    });
}

// Set up item selection events for all tabs
function setupItemSelectionEvents() {
    const itemSelectors = [
        'item-select',
        'content-item-select'
    ];
    
    itemSelectors.forEach(selectorId => {
        const selector = document.getElementById(selectorId);
        if (selector) {
            selector.addEventListener('change', function() {
                const itemId = this.value;
                if (itemId) {
                    fetchItemDetails(itemId, selectorId);
                }
            });
        }
    });
}

// Updated user recommendations function
function getUserRecommendations() {
    const userId = document.getElementById('user-select').value;
    const numRecommendations = document.getElementById('num-recommendations').value;
    const financialAwareness = document.getElementById('user-financial-awareness').value;
    
    if (!userId) {
        alert('Please select a user first');
        return;
    }
    
    console.log(`Getting recommendations for user ${userId}`);
    
    fetch(`/api/recommendations/user/${userId}?num=${numRecommendations}`)
        .then(response => response.json())
        .then(recommendations => {
            console.log(`Received ${recommendations.length} recommendations:`, recommendations);
            displayRecommendations('recommendations', recommendations, financialAwareness);
        })
        .catch(error => {
            console.error('Error getting recommendations:', error);
            const container = document.getElementById('recommendations');
            container.innerHTML = '<div class="error">Error fetching recommendations. Please try again.</div>';
        });
}

// Item-based recommendations function
function getItemRecommendations() {
    const itemId = document.getElementById('item-select').value;
    const numRecommendations = document.getElementById('item-num-recommendations').value;
    const financialAwareness = document.getElementById('item-financial-awareness').value;
    
    if (!itemId) {
        alert('Please select an item first');
        return;
    }
    
    console.log(`Getting similar items for item ${itemId}`);
    
    fetch(`/api/recommendations/item/${itemId}?num=${numRecommendations}`)
        .then(response => response.json())
        .then(recommendations => {
            console.log(`Received ${recommendations.length} recommendations:`, recommendations);
            displayRecommendations('recommendations', recommendations, financialAwareness);
        })
        .catch(error => {
            console.error('Error getting similar items:', error);
            const container = document.getElementById('recommendations');
            container.innerHTML = '<div class="error">Error fetching similar items. Please try again.</div>';
        });
}

// Get hybrid recommendations
function getHybridRecommendations() {
    const userId = document.getElementById('hybrid-user-select').value;
    const numRecommendations = document.getElementById('hybrid-num-recommendations') ? 
        document.getElementById('hybrid-num-recommendations').value : 5;
    const financialAwareness = document.getElementById('hybrid-financial-awareness').value;
    
    if (!userId) {
        alert('Please select a user first');
        return;
    }
    
    // Get the weights
    const collabWeight = parseFloat(document.getElementById('collab-weight-value').textContent);
    const contentWeight = parseFloat(document.getElementById('content-weight-value').textContent);
    const demographicWeight = parseFloat(document.getElementById('demographic-weight-value').textContent);
    
    console.log(`Getting hybrid recommendations for user ${userId} with weights: collaborative=${collabWeight}, content=${contentWeight}, demographic=${demographicWeight}`);
    
    // Build query params
    const params = new URLSearchParams({
        collaborative_weight: collabWeight,
        content_weight: contentWeight,
        contextual_weight: demographicWeight,
        num: numRecommendations
    });
    
    fetch(`/api/recommendations/hybrid/${userId}?${params.toString()}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(recommendations => {
            console.log(`Received ${recommendations.length} hybrid recommendations:`, recommendations);
            // Use the single recommendations container
            displayRecommendations('recommendations', recommendations, financialAwareness);
        })
        .catch(error => {
            console.error('Error getting hybrid recommendations:', error);
            const container = document.getElementById('recommendations');
            container.innerHTML = '<div class="error">Error fetching hybrid recommendations. Please try again.</div>';
        });
}

// Get content-based recommendations
function getContentRecommendations() {
    const itemId = document.getElementById('content-item-select').value;
    const financialAwareness = document.getElementById('content-financial-awareness').value;
    
    if (!itemId) {
        alert('Please select an item first');
        return;
    }
    
    console.log(`Getting content-based recommendations for item ${itemId}`);
    
    // Build query params for attribute weights
    const params = new URLSearchParams();
    
    // Check which attributes are selected and only include selected ones
    const categoryCheck = document.getElementById('attr-category');
    const brandCheck = document.getElementById('attr-brand');
    const tagsCheck = document.getElementById('attr-tags');
    const priceCheck = document.getElementById('attr-price');
    
    console.log('Selected attributes:', {
        category: categoryCheck && categoryCheck.checked,
        brand: brandCheck && brandCheck.checked,
        tags: tagsCheck && tagsCheck.checked,
        price: priceCheck && priceCheck.checked
    });
    
    // Only add weights for checked attributes
    if (categoryCheck && categoryCheck.checked) {
        params.append('category_weight', '0.3');
    } else {
        params.append('category_weight', '0'); // Explicitly set to 0 if unchecked
    }
    
    if (brandCheck && brandCheck.checked) {
        params.append('brand_weight', '0.2');
    } else {
        params.append('brand_weight', '0');
    }
    
    if (tagsCheck && tagsCheck.checked) {
        params.append('tags_weight', '0.4');
    } else {
        params.append('tags_weight', '0');
    }
    
    if (priceCheck && priceCheck.checked) {
        params.append('price_weight', '0.1');
    } else {
        params.append('price_weight', '0');
    }
    
    // Add number of recommendations parameter
    const numRecommendations = 10; // Default to 10 recommendations
    params.append('num', numRecommendations);
    
    const endpoint = `/api/recommendations/content/${itemId}?${params.toString()}`;
    console.log(`Making API call to: ${endpoint}`);
    
    fetch(endpoint)
        .then(response => {
            console.log(`Response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(recommendations => {
            console.log(`Received ${recommendations.length} recommendations:`, recommendations);
            // Use the single recommendations container
            displayRecommendations('recommendations', recommendations, financialAwareness);
        })
        .catch(error => {
            console.error('Error getting content recommendations:', error);
            const container = document.getElementById('recommendations');
            container.innerHTML = '<div class="error">Error fetching recommendations. Please try again.</div>';
        });
}

// Get demographic-based recommendations
function getDemographicRecommendations() {
    const userId = document.getElementById('demographic-user-select').value;
    
    if (!userId) {
        alert('Please select a user first');
        return;
    }
    
    console.log(`Getting demographic recommendations for user ${userId}`);
    
    // Get the demographic factors
    const factorAge = document.getElementById('factor-age');
    const factorGender = document.getElementById('factor-gender');
    const factorLocation = document.getElementById('factor-location');
    const factorIncome = document.getElementById('factor-income');
    const factorInterests = document.getElementById('factor-interests');
    
    // Check if elements exist before accessing checked property
    if (!factorAge || !factorGender || !factorLocation || !factorIncome || !factorInterests) {
        console.error('One or more demographic factor checkboxes not found');
        alert('Error: Demographic factor checkboxes not found. Please refresh the page.');
        return;
    }
    
    // Log raw checkbox states
    console.log('Raw checkbox states:', {
        factorAge: factorAge.checked,
        factorGender: factorGender.checked, 
        factorLocation: factorLocation.checked,
        factorIncome: factorIncome.checked,
        factorInterests: factorInterests.checked
    });
    
    const demographicFilters = {
        include_age: factorAge.checked,
        include_gender: factorGender.checked,
        include_location: factorLocation.checked,
        include_income: factorIncome.checked,
        include_interests: factorInterests.checked
    };
    
    // Get the financial awareness level
    const financialAwareness = document.getElementById('financial-awareness') ? 
        document.getElementById('financial-awareness').value : 'moderate';
        
    console.log(`Financial awareness level: ${financialAwareness}`);
    console.log('Selected demographic filters (final):', demographicFilters);
    
    // Build query params
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(demographicFilters)) {
        params.append(key, value.toString()); // Ensure boolean is converted to string
        console.log(`Adding parameter: ${key}=${value.toString()}`);
    }
    
    // Add number of recommendations parameter
    const numRecommendations = 10; // Default to 10 recommendations
    params.append('num', numRecommendations);
    
    const endpoint = `/api/recommendations/contextual/${userId}?${params.toString()}`;
    console.log(`Making API call to: ${endpoint}`);
    
    fetch(endpoint)
        .then(response => {
            console.log(`Response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(recommendations => {
            console.log(`Received ${recommendations.length} demographic recommendations:`, recommendations);
            // Display notification about which factors were used
            const factorsUsed = Object.entries(demographicFilters)
                .filter(([_, value]) => value)
                .map(([key, _]) => key.replace('include_', ''))
                .join(', ');
                
            alert(`Recommendations generated using factors: ${factorsUsed || 'none'}`);
            
            // Use the single recommendations container
            displayRecommendations('recommendations', recommendations, financialAwareness);
        })
        .catch(error => {
            console.error('Error getting demographic recommendations:', error);
            const container = document.getElementById('recommendations');
            container.innerHTML = `<div class="error">Error fetching recommendations: ${error.message}</div>`;
        });
}

// Initialize demographic factors
function initDemographicFactors() {
    // Set default values for demographic factors
    const factorAge = document.getElementById('factor-age');
    const factorGender = document.getElementById('factor-gender');
    const factorLocation = document.getElementById('factor-location');
    const factorIncome = document.getElementById('factor-income');
    const factorInterests = document.getElementById('factor-interests');
    
    // Check if elements exist and set defaults
    if (factorAge) factorAge.checked = true;
    if (factorGender) factorGender.checked = true;
    if (factorLocation) factorLocation.checked = false;
    if (factorIncome) factorIncome.checked = false;
    if (factorInterests) factorInterests.checked = true;
    
    // Add change event listeners to log changes
    const factors = [factorAge, factorGender, factorLocation, factorIncome, factorInterests];
    factors.forEach(factor => {
        if (factor) {
            factor.addEventListener('change', function() {
                console.log(`Demographic factor ${this.id} changed to ${this.checked}`);
            });
        }
    });
    
    console.log('Demographic factors initialized');
}

// NCF Recommendations
async function getNcfRecommendations() {
    const userId = document.getElementById('ncf-user-select').value;
    const numRecommendations = document.getElementById('ncf-num-recommendations').value;
    const excludeRated = document.getElementById('ncf-exclude-rated').checked;

    if (!userId) {
        showError('Please select a user');
        return;
    }

    try {
        showLoading();
        
        // Get user details and financial insights
        await Promise.all([
            fetchUserDetails(userId, 'ncf-user-select'),
            fetchFinancialInsights(userId, 'ncf-financial-info')
        ]);

        // Get NCF recommendations
        const response = await fetch(`/api/recommendations/ncf/${userId}?num=${numRecommendations}&exclude_rated=${excludeRated}`);
        const recommendations = await response.json();

        if (!response.ok) {
            throw new Error(recommendations.error || 'Failed to get recommendations');
        }

        // Get financial awareness level
        const financialAwareness = 'moderate'; // Default to moderate for NCF recommendations

        // Display recommendations in the recommendations container
        displayRecommendations('recommendations', recommendations, financialAwareness);
        
        console.log(`Displayed ${recommendations.length} NCF recommendations`);
    } catch (error) {
        console.error('Error getting NCF recommendations:', error);
        const container = document.getElementById('recommendations');
        container.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        hideLoading();
    }
}

// Train NCF Model
async function trainNcfModel() {
    const validationSplit = document.getElementById('ncf-validation-split').value;
    const epochs = document.getElementById('ncf-epochs').value;
    const batchSize = document.getElementById('ncf-batch-size').value;
    const trainingMetrics = document.querySelector('.training-metrics');
    
    try {
        // Show training metrics section
        trainingMetrics.style.display = 'block';
        
        // Disable training button
        const trainButton = document.getElementById('train-ncf-model');
        trainButton.disabled = true;
        trainButton.textContent = 'Training...';

        const response = await fetch('/api/models/ncf/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                validation_split: validationSplit,
                epochs: epochs,
                batch_size: batchSize
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to train model');
        }

        // Update metrics with final values
        document.getElementById('ncf-loss').textContent = result.history.loss[result.history.loss.length - 1].toFixed(4);
        document.getElementById('ncf-accuracy').textContent = result.history.accuracy[result.history.accuracy.length - 1].toFixed(4);
        document.getElementById('ncf-auc').textContent = result.history.auc[result.history.auc.length - 1].toFixed(4);

        showSuccess('Model training completed successfully!');
    } catch (error) {
        showError(error.message);
    } finally {
        // Re-enable training button
        const trainButton = document.getElementById('train-ncf-model');
        trainButton.disabled = false;
        trainButton.textContent = 'Train Model';
    }
}

// Loading indicator functions
function showLoading() {
    const container = document.getElementById('recommendations');
    if (container) {
        container.innerHTML = '<div class="loading">Loading recommendations...</div>';
    }
}

function hideLoading() {
    const loadingDiv = document.querySelector('.loading');
    if (loadingDiv) {
        loadingDiv.remove();
    }
} 