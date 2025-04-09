document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Load users and items on page load
    loadUsers();
    loadItems();
    
    // Set up event listeners for buttons
    document.getElementById('get-user-recommendations').addEventListener('click', getUserRecommendations);
    document.getElementById('get-item-recommendations').addEventListener('click', getItemRecommendations);
    document.getElementById('get-popular-items').addEventListener('click', getPopularItems);
});

// Fetch all users from the API
function loadUsers() {
    fetch('/users')
        .then(response => response.json())
        .then(users => {
            console.log('Loaded users:', users);
            const userSelector = document.getElementById('user-selector');
            userSelector.innerHTML = ''; // Clear existing options
            
            // Add placeholder option
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Select a user...';
            userSelector.appendChild(placeholderOption);
            
            // Add user options
            users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.user_id;
                option.textContent = `User ${user.user_id} (${user.age_group}, ${user.gender}) - ${user.city}, ${user.country}`;
                option.setAttribute('data-user', JSON.stringify(user));
                userSelector.appendChild(option);
            });

            // Add event listener to update user details when selection changes
            userSelector.addEventListener('change', updateUserDetails);
        })
        .catch(error => console.error('Error loading users:', error));
}

// Update user details display
function updateUserDetails() {
    const userSelector = document.getElementById('user-selector');
    const userDetailsElement = document.getElementById('user-details');
    
    if (userSelector.value && userDetailsElement) {
        const selectedOption = userSelector.options[userSelector.selectedIndex];
        const userData = JSON.parse(selectedOption.getAttribute('data-user'));
        
        userDetailsElement.innerHTML = `
            <div class="user-info">
                <p><strong>Location:</strong> ${userData.city}, ${userData.country}</p>
                <p><strong>Demographics:</strong> ${userData.age_group}, ${userData.gender}</p>
                <p><strong>Language:</strong> ${userData.language}</p>
                <p><strong>Income:</strong> ${userData.income_bracket}</p>
                <p><strong>Interests:</strong> ${userData.interests.join(', ')}</p>
                <p><strong>Member since:</strong> ${new Date(userData.registration_date).toLocaleDateString()}</p>
                <p><strong>Last active:</strong> ${new Date(userData.last_active).toLocaleDateString()}</p>
            </div>
        `;
        userDetailsElement.style.display = 'block';
    } else if (userDetailsElement) {
        userDetailsElement.style.display = 'none';
    }
}

// Fetch all items from the API
function loadItems() {
    fetch('/items')
        .then(response => response.json())
        .then(items => {
            console.log('Loaded items:', items);
            const itemSelector = document.getElementById('item-selector');
            itemSelector.innerHTML = ''; // Clear existing options
            
            // Add placeholder option
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Select an item...';
            itemSelector.appendChild(placeholderOption);
            
            // Group items by main category
            const categorizedItems = {};
            items.forEach(item => {
                if (!categorizedItems[item.main_category]) {
                    categorizedItems[item.main_category] = [];
                }
                categorizedItems[item.main_category].push(item);
            });
            
            // Add items by category as option groups
            for (const [category, categoryItems] of Object.entries(categorizedItems)) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = category;
                
                categoryItems.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.item_id;
                    option.textContent = `${item.name} (${item.brand})`;
                    option.setAttribute('data-item', JSON.stringify(item));
                    optgroup.appendChild(option);
                });
                
                itemSelector.appendChild(optgroup);
            }

            // Add event listener to update item details when selection changes
            itemSelector.addEventListener('change', updateItemDetails);
        })
        .catch(error => console.error('Error loading items:', error));
}

// Update item details display
function updateItemDetails() {
    const itemSelector = document.getElementById('item-selector');
    const itemDetailsElement = document.getElementById('item-details');
    
    if (itemSelector.value && itemDetailsElement) {
        const selectedOption = itemSelector.options[itemSelector.selectedIndex];
        const itemData = JSON.parse(selectedOption.getAttribute('data-item'));
        
        const stockStatus = itemData.stock_level > 20 ? 'In Stock' : 
                          itemData.stock_level > 0 ? 'Low Stock' : 
                          'Out of Stock';
        
        itemDetailsElement.innerHTML = `
            <div class="item-info">
                <p><strong>Category:</strong> ${itemData.main_category} > ${itemData.subcategory}</p>
                <p><strong>Brand:</strong> ${itemData.brand}</p>
                <p><strong>Price:</strong> £${itemData.price.toFixed(2)}</p>
                <p><strong>Condition:</strong> ${itemData.condition}</p>
                <p><strong>Rating:</strong> ${itemData.average_rating.toFixed(1)} (${itemData.num_ratings} ratings)</p>
                <p><strong>Stock:</strong> ${stockStatus} (${itemData.stock_level} units)</p>
                <p><strong>Tags:</strong> ${itemData.tags.join(', ')}</p>
                ${itemData.color ? `<p><strong>Color:</strong> ${itemData.color}</p>` : ''}
                ${itemData.size ? `<p><strong>Size:</strong> ${itemData.size}</p>` : ''}
                <p class="item-description"><strong>Description:</strong> ${itemData.description}</p>
            </div>
        `;
        itemDetailsElement.style.display = 'block';
    } else if (itemDetailsElement) {
        itemDetailsElement.style.display = 'none';
    }
}

// Get recommendations for a selected user
function getUserRecommendations() {
    const userId = document.getElementById('user-selector').value;
    if (!userId) {
        alert('Please select a user');
        return;
    }
    
    const requestData = {
        user_id: userId,
        num_recommendations: 6
    };
    
    console.log('Sending user recommendation request:', requestData);
    
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(recommendations => {
        console.log('Received recommendations:', recommendations);
        displayRecommendations(recommendations, 'user-recommendations');
    })
    .catch(error => {
        console.error('Error getting recommendations:', error);
        const container = document.querySelector('#user-recommendations .recommendation-results');
        container.innerHTML = '<p class="error">Error loading recommendations. Please try again.</p>';
    });
}

// Get similar items for a selected item
function getItemRecommendations() {
    const itemId = document.getElementById('item-selector').value;
    if (!itemId) {
        alert('Please select an item');
        return;
    }
    
    const requestData = {
        item_id: itemId,
        num_recommendations: 6
    };
    
    console.log('Sending item recommendation request:', requestData);
    
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(recommendations => {
        console.log('Received recommendations:', recommendations);
        displayRecommendations(recommendations, 'item-recommendations');
    })
    .catch(error => console.error('Error getting recommendations:', error));
}

// Get popular items
function getPopularItems() {
    const count = parseInt(document.getElementById('popular-count').value);
    
    const requestData = {
        num_recommendations: count
    };
    
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(recommendations => {
        displayRecommendations(recommendations, 'popular-recommendations');
    })
    .catch(error => console.error('Error getting popular items:', error));
}

// Display recommendations in the appropriate container
function displayRecommendations(recommendations, containerId) {
    console.log(`Displaying recommendations in ${containerId}:`, recommendations);
    const container = document.querySelector(`#${containerId} .recommendation-results`);
    
    // Clear existing content
    container.innerHTML = '';
    
    if (!recommendations || recommendations.length === 0) {
        console.log('No recommendations found');
        container.innerHTML = '<p class="placeholder">No recommendations found</p>';
        return;
    }
    
    // Create item cards for each recommendation
    recommendations.forEach((item, index) => {
        console.log(`Creating card for item ${index}:`, item);
        const itemCard = document.createElement('div');
        itemCard.className = 'item-card';
        
        // Format the score as percentage for better readability
        const scorePercent = Math.round((item.score / 5) * 100);
        
        // Create category badge class based on main category
        const categoryClass = item.main_category.toLowerCase().replace(/\s+/g, '-');
        
        // Create tags HTML
        const tagsHtml = item.tags
            ? `<div class="tags">
                ${Array.isArray(item.tags) ? item.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : ''}
               </div>`
            : '';
        
        // Create size and color info for fashion/sports items
        const sizeColorHtml = (item.main_category === 'Fashion' || item.main_category === 'Sports')
            ? `<p class="item-variant">
                ${item.color ? `<span class="color">${item.color}</span>` : ''}
                ${item.size ? `<span class="size">Size: ${item.size}</span>` : ''}
               </p>`
            : '';
        
        // Create stock level indicator
        const stockLevelClass = item.stock_level > 20 ? 'in-stock' : item.stock_level > 0 ? 'low-stock' : 'out-of-stock';
        const stockText = item.stock_level > 20 ? 'In Stock' : item.stock_level > 0 ? 'Low Stock' : 'Out of Stock';
        
        // Create HTML for the item card
        itemCard.innerHTML = `
            <div class="item-header">
                <span class="category-badge ${categoryClass}">${item.main_category}</span>
                <span class="subcategory-badge">${item.subcategory}</span>
                <h4>${item.name}</h4>
                <span class="score-badge">${scorePercent}% Match</span>
            </div>
            <div class="item-body">
                <div class="item-meta">
                    <p class="brand">${item.brand}</p>
                    <p class="condition">${item.condition}</p>
                </div>
                ${tagsHtml}
                ${sizeColorHtml}
                <p class="item-price">£${item.price.toFixed(2)}</p>
                <p class="item-description">${item.description}</p>
                <div class="item-stats">
                    <span class="rating">
                        <i class="fas fa-star"></i> ${item.average_rating.toFixed(1)}
                        <small>(${item.num_ratings} ratings)</small>
                    </span>
                    <span class="stock ${stockLevelClass}">
                        <i class="fas fa-box"></i> ${stockText}
                    </span>
                </div>
                ${item.popularity ? `<p class="item-popularity"><i class="fas fa-fire"></i> ${item.popularity} ratings</p>` : ''}
                <button class="add-to-cart" ${item.stock_level === 0 ? 'disabled' : ''}>
                    ${item.stock_level === 0 ? 'Out of Stock' : 'Add to Cart'}
                </button>
            </div>
        `;
        
        container.appendChild(itemCard);
        console.log(`Card created for item ${index}`);
    });
    
    // Add event listeners to add-to-cart buttons
    document.querySelectorAll('.add-to-cart').forEach(button => {
        if (!button.disabled) {
            button.addEventListener('click', function() {
                const productName = this.closest('.item-card').querySelector('h4').textContent;
                alert(`Added ${productName} to cart! (Demo functionality)`);
            });
        }
    });
} 