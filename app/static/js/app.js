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
                option.textContent = `${user.name} (${user.age}, ${user.gender}) - ${user.city}`;
                option.setAttribute('data-interests', user.interests);
                option.setAttribute('data-spending', user.spending_profile);
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
        const interests = selectedOption.getAttribute('data-interests');
        const spending = selectedOption.getAttribute('data-spending');
        const spendingStars = '★'.repeat(parseInt(spending)) + '☆'.repeat(5 - parseInt(spending));
        
        userDetailsElement.innerHTML = `
            <div class="user-interests"><strong>Interests:</strong> ${interests}</div>
            <div class="user-spending"><strong>Spending Profile:</strong> ${spendingStars}</div>
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
            const itemSelector = document.getElementById('item-selector');
            itemSelector.innerHTML = ''; // Clear existing options
            
            // Add placeholder option
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'Select an item...';
            itemSelector.appendChild(placeholderOption);
            
            // Group items by category
            const categorizedItems = {};
            items.forEach(item => {
                if (!categorizedItems[item.category]) {
                    categorizedItems[item.category] = [];
                }
                categorizedItems[item.category].push(item);
            });
            
            // Add items by category as option groups
            for (const [category, categoryItems] of Object.entries(categorizedItems)) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = category;
                
                categoryItems.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.item_id;
                    option.textContent = `${item.name} (${item.brand}, £${item.price.toFixed(2)})`;
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
    const itemId = document.getElementById('item-selector').value;
    const itemDetailsElement = document.getElementById('item-details');
    
    if (itemId && itemDetailsElement) {
        fetch(`/items`)
            .then(response => response.json())
            .then(items => {
                const item = items.find(i => i.item_id == itemId);
                if (item) {
                    itemDetailsElement.innerHTML = `
                        <div class="item-category"><strong>Category:</strong> ${item.category}</div>
                        <div class="item-brand"><strong>Brand:</strong> ${item.brand}</div>
                        <div class="item-price"><strong>Price:</strong> £${item.price.toFixed(2)}</div>
                        <div class="item-description">${item.description}</div>
                    `;
                    itemDetailsElement.style.display = 'block';
                }
            });
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
    
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(recommendations => {
        displayRecommendations(recommendations, 'user-recommendations');
    })
    .catch(error => console.error('Error getting recommendations:', error));
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
    
    fetch('/get_recommendations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(recommendations => {
        displayRecommendations(recommendations, 'item-recommendations');
    })
    .catch(error => console.error('Error getting similar items:', error));
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
    const container = document.querySelector(`#${containerId} .recommendation-results`);
    
    // Clear existing content
    container.innerHTML = '';
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="placeholder">No recommendations found</p>';
        return;
    }
    
    // Create item cards for each recommendation
    recommendations.forEach(item => {
        const itemCard = document.createElement('div');
        itemCard.className = 'item-card';
        
        // Format the score as percentage for better readability
        const scorePercent = Math.round((item.score / 5) * 100);
        
        // Create category badge class based on category
        const categoryClass = item.category.toLowerCase().replace(/\s+/g, '-');
        
        // Create HTML for the item card
        itemCard.innerHTML = `
            <div class="item-header">
                <span class="category-badge ${categoryClass}">${item.category}</span>
                <h4>${item.name}</h4>
                <span class="score-badge">${scorePercent}% Match</span>
            </div>
            <div class="item-body">
                <p class="item-meta">${item.brand}</p>
                <p class="item-price">£${item.price.toFixed(2)}</p>
                <p class="item-description">${item.description}</p>
                ${item.popularity ? `<p class="item-popularity"><strong>Popularity:</strong> ${item.popularity} ratings</p>` : ''}
                <button class="add-to-cart">Add to Cart</button>
            </div>
        `;
        
        container.appendChild(itemCard);
    });
    
    // Add event listeners to add-to-cart buttons
    document.querySelectorAll('.add-to-cart').forEach(button => {
        button.addEventListener('click', function() {
            const productName = this.closest('.item-card').querySelector('h4').textContent;
            alert(`Added ${productName} to cart! (Demo functionality)`);
        });
    });
} 