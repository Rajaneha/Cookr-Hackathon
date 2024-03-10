# Cookr-Hackathon
# Problem Statement1
# Item Categorization 

## Description
Create a model or research the necessary steps to categorize items automatically. When a cook adds an item to their kitchen, it should be categorized into multiple predefined categories. Sample data will be provided for training.

### Example Categories
- Idly: South Indian, Protein Rich, Breakfast, Baked Items, etc.
- Chicken Vindaloo: North Indian, Punjabi, Non-Veg, Chicken, Protein Rich, etc.
- Ragi Dosa: South Indian, Diabetic Friendly, Millet Based, Pregnancy Friendly, etc.
  
# Problem Statement2
# Last Mile Delivery Batching 
## Description
Optimize last-mile delivery for speed and cost efficiencies using smarter algorithms in the ecommerce marketplace. Group/batch the delivery of multiple items to the same rider without losing time.

### Operational Research Algorithms
#### Rule #1:
- Two orders - From the same kitchen.
- To the same customer.
- Ready at the same time (10 mins apart).
- Assign the pick-up to the same rider.
#### Rule #2:
- Two orders.
- From two different kitchens (1 km apart).
- To the same customer.
- Ready at the same time (10 mins apart).
- Assign the pick-up to the same rider.
#### Rule #3:
- Two orders.
- From the same kitchen.
- To two different customers (1 km apart).
- Ready at the same time (10 mins apart).
- Assign the pick-up to the same rider.
#### Rule #4:
- Two orders.
- From two different kitchens (1 km apart).
- To the same customer.
- Ready at the same time (10 mins apart).
- Assign the pick-up to the same rider.

#### Rule #5:

- Two orders.
- From two different kitchens (1 km apart).
- To the same customer.
- Ready at the sam#### Rule #6:

#### Rule #6:
- Two orders.
- To the same customer.
- 2nd kitchen's pick-up on the way to the customer.
- Ready at the time the rider reaches the second kitchen (10 mins apart).
- Assign the pick-up to the same rider.

#### Rule #7:

- Two orders.
- 2nd customer's drop on the way to the 1st customer (Vice Versa).
- 2nd kitchen's pick-up on the way to the customer.
- Ready at the same time (10 mins apart or by the time rider reaches the kitchen).
- Assign the pick-up to the same rider.

#### Rule #8:

- Two orders.
- From the same kitchen.
- 2nd customer's drop on the way to the customer 1st (Vice Versa).
- Ready at the same time (10 mins apart).
- Assign the pick-up to the same rider
