from torch.autograd import Variable
import numpy as np
import itertools


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    
    lossHist = {}
    lossHist['trainLoss'] = [] 
    lossHist['valLoss'] = []
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        
        lossHist['trainLoss'].append(train_loss)
        lossHist['valLoss'].append(val_loss)
        
        print(message)
    return lossHist

def fit2(train_loader_primary, val_loader_primary, train_loader_helper, val_loader_helper, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, mode='contrastive'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    
    lossHist = {}
    lossHist['trainLoss'] = [] 
    lossHist['valLoss'] = []
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch2(train_loader_primary, train_loader_helper, model, loss_fn, optimizer, cuda, log_interval, metrics, mode)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch2(val_loader_primary, val_loader_helper, model, loss_fn, cuda, metrics, mode)
        
        # what about helper
        val_loss /= len(val_loader_primary)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        
        lossHist['trainLoss'].append(train_loss)
        lossHist['valLoss'].append(val_loss)
        
        print(message)
    return lossHist



def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = Variable(target)
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.data[0])
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


# Alternative for two different embeddings
def train_epoch2(train_loader_primary, train_loader_helper, model, loss_fn, optimizer, cuda, log_interval, metrics, mode='contrastive'):
    
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    
    #for id, (a, b) in enumerate(itertools.izip(as, bs)):

    for batch_idx, ((data_primary, target_primary), (data_helper, target_helper)) in enumerate(itertools.izip(train_loader_primary, train_loader_helper)):

        # primary domain
        target_primary = target_primary if len(target_primary) > 0 else None
        #if not type(data_primary) in (tuple, list):
        #    data_primary = (data_primary,)
        if cuda:
            #data_primary = tuple(d.cuda() for d in data_primary)
            data_primary = data_primary.cuda()
            if target_primary is not None:
                target_primary = target_primary.cuda()
        data_primary = Variable(data_primary)
        #data_primary = tuple(Variable(d) for d in data_primary)

        #print data_primary
        # primary should get the output from the same
        #outputs_primary = model(*data_primary)


        
        # helper domain
        target_helper = target_helper if len(target_helper) > 0 else None
        #if not type(data_helper) in (tuple, list):
        #    data_helper = (data_helper,)
        if cuda:
            
            data_helper = data_helper.cuda()
            #data_helper = tuple(d.cuda() for d in data_helper)
            if target_helper is not None:
                target_helper = target_helper.cuda()
        data_helper = Variable(data_helper)

        #data_helper = tuple(Variable(d) for d in data_helper)

        
        # primary should get the output from the same
        #print type(data_primary)
        #print type(data_helper)
        if mode=='contrastive':
            outputs_primary, outputs_helper = model(data_primary, data_helper)
        elif mode=='triplet':
            outputs_primary, outputs_helper, output_helper = model(data_primary, data_helper, data_helper)
        
        if type(outputs_primary) not in (tuple, list):
            outputs_primary = (outputs_primary,)
        if type(outputs_helper) not in (tuple, list):
            outputs_helper = (outputs_helper,)
        
        optimizer.zero_grad()
            
        loss_inputs = outputs_primary
        if target_primary is not None:
            target_primary = Variable(target_primary)
            target_primary = (target_primary,)
            loss_inputs += target_primary
            
        loss_inputs += outputs_helper
        if target_helper is not None:
            target_helper = Variable(target_helper)
            target_helper = (target_helper,)
            loss_inputs += target_helper

        # calculate the online loss functions
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.data[0])
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        # what about metric ??
        for metric in metrics:
            metric(outputs_primary, target_primary, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data_primary[0]), len(train_loader_primary.dataset),
                100. * batch_idx / len(train_loader_primary), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics





def test_epoch2(val_loader_primary, val_loader_helper, model, loss_fn, cuda, metrics, mode='contrastive'):
    for metric in metrics:
        metric.reset()
    model.eval()
    val_loss = 0
    
    for batch_idx, ((data_primary, target_primary), (data_helper, target_helper)) in enumerate(itertools.izip(val_loader_primary, val_loader_helper)):
        
        # primary domain
        target_primary = target_primary if len(target_primary) > 0 else None
        #if not type(data_primary) in (tuple, list):
        #    data_primary = (data_primary,)
        if cuda:
            data_primary = data_primary.cuda()
            #data_primary = tuple(d.cuda() for d in data)
            if target_primary is not None:
                target_primary = target_primary.cuda()
        #data_primary = tuple(Variable(d, volatile=True) for d in data_primary)
        data_primary = Variable(data_primary)
        
        # helper domain
        target_helper = target_helper if len(target_helper) > 0 else None
        #if not type(data_helper) in (tuple, list):
        #    data_helper = (data_helper,)
        if cuda:
            data_helper = data_helper.cuda()
            #data_helper = tuple(d.cuda() for d in data)
            if target_helper is not None:
                target_helper = target_helper.cuda()
        #data_helper = tuple(Variable(d, volatile=True) for d in data_helper)
        data_helper = Variable(data_helper)
        
        if mode=='contrastive':
            outputs_primary, outputs_helper = model(data_primary, data_helper)
        elif mode=='triplet':
            outputs_primary, outputs_helper, output_helper = model(data_primary, data_helper, data_helper)
        
        
        if type(outputs_primary) not in (tuple, list):
            outputs_primary = (outputs_primary,)
        if type(outputs_helper) not in (tuple, list):
            outputs_helper = (outputs_helper,)            
            
        loss_inputs = outputs_primary
        if target_primary is not None:
            target_primary = Variable(target_primary, volatile=True)
            target_primary = (target_primary,)
            loss_inputs += target_primary
            


        loss_inputs += outputs_helper
        if target_helper is not None:
            target_helper = Variable(target_helper, volatile=True)
            target_helper = (target_helper,)
            loss_inputs += target_helper

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        val_loss += loss.data[0]

        # what about metrices ??
        for metric in metrics:
            metric(outputs_primary, target_primary, loss_outputs)

    return val_loss, metrics
